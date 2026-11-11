"""The native MuJoCo simulator (`--backend mujoco`): the robot of `robot.py` (UR5e on the pedestal and slide, L tool with the
suction cup, flange force sensor) in the cabinet and scene of the task, with primitives written for it.

Shared with the other backends (so a seed gives the same scene, and the learning code sees the same observations): the scene
sampling and the task logic of `l2r.sim.scene.Scene` (recorded by `MujocoClient`, which also answers its pose queries), the camera
and the observation pipeline. Native: the robot model, the controller (plan-then-execute line moves, collision checked; see
`control.py`), the suction (an adhesion actuator, with a force limit) and the primitives below.

Primitives (same arguments as `l2r.sim.primitives.Manipulator`; frames: x the cabinet depth, the robot at +x, y the width, z up):
* push: the tip goes to the start point and moves `push_length` along y (sign of sin(angle)) or along the floor direction
  `heading`. The blade points along the push when that line enters through the opening (joint 6 turns it), else straight in.
* pick: in along the approach line (rotated by `approach` about the vertical) to the object's face at PICK_HEIGHT, stop on cup
  contact, vacuum on, lift 2 cm, out along the line; the arm then waits at the carry pose (the tip 20 cm in front of the opening)
  until the place.
* place: the held object carried to the point that puts it at `xy`, vacuum off, out.
* pull: attach as a pick, drag `distance` along `heading` on the floor, vacuum off, out.
Every primitive starts and ends at the home posture outside the cabinet (a pick ends at the carry pose, the place starts there); a line that cannot be followed without collisions is
refused before the robot moves (the primitive then reports failure and the scene is untouched)."""
import logging

import mujoco
import numpy as np

from l2r.config import SimParams
from l2r.perception import make_perception
from l2r.sim.environment import TIME_STEP, Environment
from l2r.sim.mj.camera import MujocoCamera
from l2r.sim.mj.client import MujocoClient
from l2r.sim.mj.control import Controller, Kinematics
from l2r.sim.mj.robot import BASE_POSITION, ROBOT as ROBOT_BIT, SCENE as SCENE_BIT, STAGE_MARGIN, ToolGeometry, robot_spec
from l2r.sim.primitives import MAX_APPROACH, OPENING_MARGIN, PICK_HEIGHT, PUSH_HEIGHT, WALL_MARGIN
from l2r.sim.scene import Scene
from l2r.sim.workspace import Workspace

log = logging.getLogger(__name__)

FORCE_LIMIT = 80.0  # N at the flange: a motion stops above it (the tool weighs ~6 N; a push of a light object ~3 N)
SPEED = 0.15  # m/s in free space
PUSH_SPEED = 0.05  # m/s while pushing / approaching an object
SETTLE_TICKS = 25
LIFT = 0.02  # m: a grasped object is lifted this much off the floor
APPROACHES = tuple(np.radians([0, 15, -15, 30, -30, 45, -45]))  # tool headings tried (as l2r.envs.approach)
HELD_PENETRATION = 0.003  # m: a held object counts as colliding with the cabinet only beyond this penetration
CARRY_MARGIN = 0.2  # m: the tip this far in front of the opening while an object is held (out of the cabinet)


def horizontal(heading):
    """Tip axes for a tool pointing horizontally into the cabinet along (-cos h, -sin h, 0); its 'x' axis down."""
    z = np.array([-np.cos(heading), -np.sin(heading), 0.0])
    x = np.array([0, 0, -1.0])
    return np.column_stack([x, np.cross(z, x), z])


class Suction:
    """The vacuum cup. Switched on while the cup touches an object, it welds that object to the cup where it is (a stiff
    equality constraint); the weld lets go when the force it transmits exceeds `max_force` for two control ticks in a row (the
    vacuum's holding force: a real 40 mm cup at -0.8 bar holds about 100 N), and when the vacuum is switched off.

    (MuJoCo's adhesion actuator was tried first: its contacts carry no friction while the surfaces are within the margin but
    apart, so it pulled objects in but could not lift them.)"""
    is_gripper = False

    def __init__(self, max_force=80.0):
        self.max_force = max_force
        self.model = self.data = None
        self.activated = False
        self.grip_body = None  # the PyBullet-style id of the held object
        self.offset = None  # held object position - tip position, at the grasp (world frame)
        self.peeled = 0  # holds lost to the force limit (diagnostics)
        self._over = 0

    def bind(self, model, data, body_of, welds):
        """welds: {object id: equality index} of the cup welds."""
        self.model, self.data, self.body_of, self.welds = model, data, body_of, welds
        self.cup = model.geom('cup').id
        self.cup_body = model.body('cup').id
        self.activated, self.grip_body, self.offset, self._weld = False, None, None, None

    def touching(self, owner):
        """PyBullet-style ids of the objects the cup touches (contacts within 2 mm)."""
        found = set()
        for c in self.data.contact[:self.data.ncon]:
            if self.cup in (c.geom1, c.geom2) and c.dist < 0.002:
                uid = int(owner[c.geom2 if c.geom1 == self.cup else c.geom1])
                if uid >= 0:
                    found.add(uid)
        return found

    def activate(self, uid, tip):
        """Weld object `uid` to the cup at its current relative pose."""
        m, d = self.model, self.data
        e = self.welds[uid]
        cup, body = self.cup_body, self.body_of[uid]
        rel_pos = d.xmat[cup].reshape(3, 3).T @ (d.xpos[body] - d.xpos[cup])
        rel_quat, inv = np.zeros(4), np.zeros(4)
        mujoco.mju_negQuat(inv, d.xquat[cup])
        mujoco.mju_mulQuat(rel_quat, inv, d.xquat[body])
        m.eq_data[e, :3] = 0.0
        m.eq_data[e, 3:6] = rel_pos
        m.eq_data[e, 6:10] = rel_quat
        m.eq_data[e, 10] = 1.0
        d.eq_active[e] = 1
        self.activated, self.grip_body, self._weld, self._over = True, uid, e, 0
        self.offset = d.xipos[body] - tip

    def release(self):
        if self.data is not None and self._weld is not None:
            self.data.eq_active[self._weld] = 0
        self.activated, self.grip_body, self.offset, self._weld = False, None, None, None

    def force(self):
        """Magnitude of the force the weld transmits (N)."""
        d = self.data
        rows = np.nonzero((d.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY) & (d.efc_id == self._weld))[0]
        return float(np.linalg.norm(d.efc_force[rows[:3]])) if len(rows) else 0.0

    def check(self):
        """After every control tick: let go if the weld carried more than the vacuum holds."""
        if not self.activated:
            return
        self._over = self._over + 1 if self.force() > self.max_force else 0
        if self._over >= 2:
            log.debug('suction: let go of %s at %.0f N', self.grip_body, self.force())
            self.peeled += 1
            self.release()


class NativeEnvironment(Environment):
    def __init__(self, gui=False, time_step=TIME_STEP, num_obj=(9,), params=None, perception=None, tool=None, base=None):
        if gui:
            raise NotImplementedError('MuJoCo backend: no GUI (render with the camera or mujoco.viewer)')
        self.params = params or SimParams()
        if self.params.end_effector == 'gripper':
            raise NotImplementedError('MuJoCo backend: suction cup only')
        self.num_obj, self.time_step = list(num_obj), time_step
        self.ws = Workspace.from_params(self.params)
        self.perception = perception or make_perception(self.params)
        self.p = MujocoClient()
        self.scene = Scene(self.p, self.params, self.ws)
        self.camera = MujocoCamera(self.p, self.params, self.ws)
        blade = getattr(self.params, 'mj_blade', None)
        self.tool_geometry = tool or (ToolGeometry(blade=blade) if blade else ToolGeometry())
        mount = getattr(self.params, 'mj_base', None)
        self.base = base or ((mount[0], 0.0, mount[1]) if mount else BASE_POSITION)
        self.suction = self.tool = Suction()
        self.kin = self.ctl = None
        self._force_stops, self._force_events = 0, []
        self.last_refused = False  # the last primitive was refused before the robot moved (infeasible), see `info['refused']`

    # ------------------------------------------------------------------ episode
    def reset(self):
        """Same random draws as the PyBullet environment (camera, regions, objects), then one compiled world with the robot at
        its home posture; the objects settle."""
        p, params = self.p, self.params
        self.camera.randomize()
        p.resetSimulation()
        p.setTimeStep(self.time_step)
        if params.solver_iterations:
            p.setPhysicsEngineParameter(numSolverIterations=int(params.solver_iterations))
        p.setGravity(0, 0, -9.81)
        p.loadURDF('plane.urdf', basePosition=(0, 0, 0), useFixedBase=True)
        self.scene.build_cabinet()
        self.scene.spawn_objects(self.num_obj)
        p.compile(base_spec=robot_spec(self.tool_geometry, base=self.base), settle=self._place_robot, weld_to='cup')
        self.tool.bind(p.model, p.data, p.body_of, p.weld_of)
        self.ctl.on_tick = self._tick
        self._jac = np.zeros((3, p.model.nv))
        self.wait_static()
        return self.scene.is_valid()

    def _place_robot(self, model, data):
        self.kin = Kinematics(model)
        home, _ = self.kin.solve(self.kin.home, self._staging(0.0, 0.0), horizontal(0.0))
        self.kin.home = home
        data.qpos[self.kin.qadr] = home
        self.ctl = Controller(model, data, self.kin)
        self.ctl.command = home.copy()
        data.ctrl[self.ctl.acts] = home
        self._free = _collision_check(model, self.kin)

    def _tick(self):
        """After every control tick: the suction's force check, and the held object's weight compensated in the arm joints
        (the payload setting of a real UR controller), so that the position servos track with an object on the cup."""
        tool, d, m = self.tool, self.p.data, self.p.model
        tool.check()
        d.qfrc_applied[self.kin.dofs] = 0.0
        if tool.activated and tool.grip_body is not None:
            body = self.p.body_of[tool.grip_body]
            mujoco.mj_jac(m, d, self._jac, None, d.xipos[body], tool.cup_body)
            weight = -m.body_mass[body] * m.opt.gravity  # the force the arm must supply
            d.qfrc_applied[self.kin.dofs] = (self._jac.T @ weight)[self.kin.dofs]

    # ------------------------------------------------------------------ geometry
    def _staging(self, y, heading):
        """Tip position in front of the opening on the line through (front, y) at `heading`, at push height."""
        b = self.ws.bounds
        return np.array([b[0][1] + STAGE_MARGIN, y, b[2][0] + PUSH_HEIGHT])

    def _line_start(self, target, heading):
        """Point in front of the opening on the line into the cabinet through `target` at `heading`, or None if the line
        does not pass the opening (or is steeper than MAX_APPROACH)."""
        if abs(heading) > MAX_APPROACH:
            return None
        b = self.ws.bounds
        direction = horizontal(heading)[:, 2]
        front = b[0][1] + STAGE_MARGIN
        start = np.asarray(target, float) - direction * (front - target[0]) / -direction[0]
        entry_y = target[1] - direction[1] * (b[0][1] - target[0]) / -direction[0]
        if not (b[1][0] + OPENING_MARGIN <= entry_y <= b[1][1] - OPENING_MARGIN):
            return None
        return start

    # ------------------------------------------------------------------ motion helpers
    def _free_now(self, q, strict=False):
        held = self.tool.grip_body if self.tool.activated else None
        return self._free(q, self.p.data, None if held is None else self.p.body_of[held], strict)

    def _free_strict(self, q):
        return self._free_now(q, strict=True)

    def _go(self, position, rotation):
        """To a tip pose outside the cabinet. Holding an object: Cartesian lines (sideways in front of the opening, then to the
        pose), checked every centimetre with the object; otherwise a joint move, checked every 0.05 rad."""
        if self.holds():
            tip = self.ctl.tip()[0]
            side = np.array([max(tip[0], position[0]), position[1], position[2]])
            return (self._line(side, rotation, SPEED, force=False).reached and
                    self._line(position, rotation, SPEED, force=False).reached)
        q, ok = self.kin.solve(self.ctl.measured(), position, rotation)
        if not ok:
            return False
        start = self.ctl.measured()
        samples = max(5, int(np.ceil(np.abs(q - start).max() / 0.05)))
        if not all(self._free_now(start + (q - start) * s) for s in np.linspace(1.0 / samples, 1.0, samples)):
            return False
        return self.ctl.move_joints(q).reached

    def _home(self):
        return self.ctl.move_joints(self.kin.home).reached

    def _line(self, end, rotation, speed, force=True, stop=None):
        result = self.ctl.move_line(end, rotation, speed=speed, force_limit=FORCE_LIMIT if force else None, stop=stop,
                                    collision_free=self._free_now)
        if not result.reached:
            log.debug('line to %s: %s', np.round(end, 3), 'force stop (%.0f N)' % result.max_force if result.force_stop else
                      ('refused (no collision-free plan: %s)' % (self._free.last,) if result.steps == 0 else 'not reached in time'))
        if result.force_stop:
            self._force_stops += 1
            self._force_events.append((float(self.ctl.tip()[0][0]), float(result.max_force)))
        return result

    def _leave(self, rotation):
        """From wherever the tip is: back out along the tool axis until it is 5 cm in front of the opening, then home."""
        tip = self.ctl.tip()[0]
        axis = rotation[:, 2]  # pointing into the cabinet (negative x)
        back = max(self.ws.bounds[0][1] + STAGE_MARGIN + 0.05 - tip[0], 0.0) / max(-axis[0], 1e-3)
        if back > 0:
            self._line(tip - axis * back, rotation, SPEED, force=False)
        return self._home()

    # ------------------------------------------------------------------ primitives
    def push(self, xy, angle=0.0, push_length=0.08, speed=0.01, heading=None):
        self.last_refused = False
        b = self.ws.bounds
        start = np.array([xy[0], xy[1], b[2][0] + PUSH_HEIGHT])
        if heading is None:
            direction = np.array([0.0, np.sign(np.sin(angle)) or 1.0, 0.0])
        else:
            direction = np.array([np.cos(heading), np.sin(heading), 0.0])
            if direction[0] > 1e-6:
                raise ValueError('a push cannot move towards the robot (heading %.2f rad): use pull' % heading)
        end = start + push_length * direction
        end[0] = max(end[0], b[0][0] + WALL_MARGIN)
        end[1] = min(max(end[1], b[1][0] + WALL_MARGIN), b[1][1] - WALL_MARGIN)
        tool_heading = 0.0  # the blade along the push when that line enters through the opening, else straight in
        if heading is not None and self._line_start(start, heading - np.pi) is not None:
            tool_heading = float(np.arctan2(np.sin(heading - np.pi), np.cos(heading - np.pi)))
        rotation = horizontal(tool_heading)
        entry = self._line_start(start, tool_heading)
        if entry is None or not self._go(entry, rotation):
            log.debug('push %s: %s', np.round(start, 3), 'no line through the opening' if entry is None else 'no joint move to the entry')
            self._home()
            self.last_refused = True
            return False
        if self.ctl.plan_line(start, rotation, collision_free=self._free_now) is None:
            log.debug('push %s: the approach line is not collision free', np.round(start, 3))
            self._home()
            self.last_refused = True
            return False
        approached = self._line(start, rotation, SPEED)
        pushed = approached.reached and self._line(end, rotation, PUSH_SPEED).reached
        self._leave(rotation)
        return bool(pushed) and self.scene.is_valid()

    def _attach(self, xy, approach, height):
        """In along the approach line to the object at `xy`, stop on cup contact, vacuum on. (attached, rotation, entry) or None
        (refused). If the requested approach cannot be followed (near a side wall the wrist would hit it), the other approach
        angles are tried, those coming from the cabinet's middle towards the wall first (they keep the wrist off the wall)."""
        b = self.ws.bounds
        target = np.array([xy[0], xy[1], b[2][0] + (height if height is not None else PICK_HEIGHT)])
        towards_wall = np.sign(-xy[1]) if abs(xy[1]) > 1e-6 else 1.0  # an approach angle of this sign comes from the middle
        others = sorted((a for a in APPROACHES if abs(a - approach) > 1e-6),
                        key=lambda a: (np.sign(a) != towards_wall and a != 0.0, abs(a)))
        chosen = None
        for angle in [approach] + others:
            rotation = horizontal(angle)
            entry = self._line_start(target, angle)
            if entry is None or not self._feasible(entry, target, rotation):
                continue
            chosen = (angle, rotation, entry)
            break
        if chosen is None:
            log.debug('attach %s: no feasible approach line', np.round(target, 3))
            self.last_refused = True
            return None
        angle, rotation, entry = chosen
        if angle != approach:
            log.debug('attach %s: approach %.0f deg instead of %.0f', np.round(target, 3), np.degrees(angle), np.degrees(approach))
        if not self._go(entry, rotation) or self.ctl.plan_line(target, rotation, collision_free=self._free_now) is None:
            log.debug('attach %s: the approach line is not collision free', np.round(target, 3))
            self._home()
            self.last_refused = True
            return None
        tool = self.tool
        tool.release()
        owner = self.p.geom_owner
        touched = lambda: bool(tool.touching(owner))
        end = target + rotation[:, 2] * 0.04  # up to 4 cm past the face point, but not into the back wall
        end[0] = max(end[0], self.ws.bounds[0][0] + self.tool_geometry.cup_radius + 0.005)
        self._line(end, rotation, PUSH_SPEED, stop=touched)
        objects = tool.touching(owner)
        if not objects:
            log.debug('attach %s: no cup contact (tip stopped at %s)', np.round(target, 3), np.round(self.ctl.tip()[0], 3))
            return False, rotation, entry
        held = min(objects, key=lambda uid: np.linalg.norm(self.p.data.xipos[self.p.body_of[uid]] - self.ctl.tip()[0]))
        tool.activate(held, self.ctl.tip()[0])
        self.ctl.hold(SETTLE_TICKS)
        return tool.activated, rotation, entry

    def pick(self, xy, speed=0.01, yaw=0.0, approach=0.0, height=None):
        self.last_refused = False
        result = self._attach(xy, approach, height)
        if result is None:
            return False
        attached, rotation, entry = result
        if attached:
            tip = self.ctl.tip()[0]
            self._line(tip + np.array([0, 0, LIFT]), rotation, PUSH_SPEED, force=False)  # off the floor
            tip = self.ctl.tip()[0]
            axis = rotation[:, 2]
            out = tip - axis * max(self.ws.bounds[0][1] + CARRY_MARGIN - tip[0], 0.0) / max(-axis[0], 1e-3)
            self._line(out, rotation, SPEED, force=False)  # out of the cabinet along the approach line
            if self.holds():  # the carry pose: in front of the opening (within its width), the tool straight
                b = self.ws.bounds
                carry_y = float(np.clip(out[1], b[1][0] + OPENING_MARGIN, b[1][1] - OPENING_MARGIN))
                self._go(np.array([b[0][1] + CARRY_MARGIN, carry_y, out[2]]), horizontal(0.0))
        held = attached and self.holds()
        if not held:
            self.tool.release()
            self._leave(rotation)
        return bool(held) and self.scene.is_valid(ignore=(self.tool.grip_body,) if held else ())

    def pull(self, xy, approach=0.0, heading=0.0, distance=0.08, speed=0.01, height=None):
        self.last_refused = False
        b = self.ws.bounds
        result = self._attach(xy, approach, height)
        if result is None:
            return False
        attached, rotation, entry = result
        dragged = False
        if attached:
            tip = self.ctl.tip()[0]
            end = tip + distance * np.array([np.cos(heading), np.sin(heading), 0.0])
            end[0] = min(end[0], b[0][1] - WALL_MARGIN)
            end[1] = min(max(end[1], b[1][0] + WALL_MARGIN), b[1][1] - WALL_MARGIN)
            dragged = self._line(end, rotation, PUSH_SPEED).reached
        self.tool.release()
        self.ctl.hold(SETTLE_TICKS)
        self._leave(rotation)
        return bool(attached and dragged) and self.scene.is_valid()

    def place(self, xy, speed=0.01):
        """Put the held object at `xy`. Options: the requested point, then points shallower and / or further from the side wall
        (up to 12 cm and 9 cm: the regions are bands along y, a shift away from the wall keeps the object in its region), each
        with the tool straight in or, if that is infeasible, turned by up to 45 degrees (the object turns with it). The nearest
        feasible point is taken. Executed (True) only at the requested point; never dropped outside when a point inside is
        reachable."""
        self.last_refused = False
        b, tool = self.ws.bounds, self.tool
        offset = tool.offset if tool.offset is not None else np.zeros(3)
        carry_z = self.ctl.tip()[0][2]  # in at the carry height (the object clear of the floor), lowered at the end
        inward = -np.sign(xy[1]) if abs(xy[1]) > 1e-6 else 1.0  # away from the nearer side wall
        shifts = sorted(((dx, dy) for dx in (0.0, 0.03, 0.06, 0.09, 0.12) for dy in (0.0, 0.03, 0.06, 0.09)),
                        key=lambda shift: np.hypot(*shift))
        options = []
        for dx, dy in shifts:
            for heading in APPROACHES:
                turn = np.array([[np.cos(heading), -np.sin(heading), 0], [np.sin(heading), np.cos(heading), 0], [0, 0, 1]])
                tip_xy = np.array([xy[0] + dx, xy[1] + inward * dy]) - (turn @ offset)[:2]  # the object, turned with the tool
                position = np.array([tip_xy[0], tip_xy[1], carry_z])
                if position[0] > b[0][1] - 0.02:
                    continue
                entry = self._line_start(position, heading)
                if entry is not None:
                    options.append((dx, dy, heading, position, entry))
        chosen = None
        by_point = {}
        for option in options:
            by_point.setdefault(option[:2], []).append(option)
        for point in sorted(by_point, key=lambda shift: np.hypot(*shift)):  # the nearest point first; there straight in, else
            # the smallest turn that is feasible (turning to keep the object off its neighbours made toppling more frequent:
            # 0.84 / 0.74 against 0.87 / 0.79 at 5 / 6 objects)
            chosen = next((o for o in by_point[point] if self._feasible(o[4], o[3], horizontal(o[2]))), None)
            if chosen is not None:
                break
        placed, done = False, False
        if chosen is not None:
            dx, dy, heading, position, entry = chosen
            rotation = horizontal(heading)
            if self._go(entry, rotation) and self.ctl.plan_line(position, rotation, collision_free=self._free_now) is not None:
                placed = self._line(position, rotation, SPEED).reached and dx == 0.0 and dy == 0.0
                self._line(position - np.array([0, 0, LIFT - 0.005]), rotation, PUSH_SPEED, force=False)  # set it down
                done = True
                if dx or dy or heading:
                    log.debug('place %s: %.2f m shallower, %.2f m from the wall, tool at %.0f deg', np.round(xy, 3), dx, dy,
                              np.degrees(heading))
        if not done:
            log.debug('place %s: no reachable point nearby', np.round(xy, 3))
            self.last_refused = True
        rotation = horizontal(chosen[2]) if chosen is not None else horizontal(0.0)
        tool.release()
        self.ctl.hold(SETTLE_TICKS)
        self._leave(rotation)
        return bool(placed) and self.scene.is_valid()

    def _feasible(self, entry, target, rotation, check=None):
        """Before moving: the entry pose and the end pose have collision-free IK solutions (from the current posture) and the
        line between them can be planned from the entry."""
        check = check or self._free_now
        q_entry, ok = self.kin.solve(self.kin.home, entry, rotation)
        if not ok or not check(q_entry):
            return False
        q_end, ok = self.kin.solve(q_entry, target, rotation)
        return bool(ok and check(q_end))

    def holds(self):
        """The cup still holds its object (the weld has not let go)."""
        return bool(self.tool.activated and self.tool.grip_body is not None)

    # ------------------------------------------------------------------ validity / diagnostics
    def is_valid(self):
        held = self.tool.grip_body if self.holds() else None
        return self.scene.is_valid(ignore=() if held is None else (held,))

    @property
    def force_stops(self):
        return self._force_stops

    @property
    def force_events(self):
        return self._force_events


def _collision_check(model, kin):
    """free(q, data, attached): no contact of the robot with the cabinet, the pedestal or itself, and none of the `attached`
    (held) body with the cabinet (contacts with the other movable objects are allowed: pushing them is the point)."""
    probe = mujoco.MjData(model)
    movable = {b for b in range(model.nbody) if model.body_jntnum[b] and
               model.jnt_type[model.body_jntadr[b]] == mujoco.mjtJoint.mjJNT_FREE}

    def name(g):
        return model.geom(g).name or model.body(model.geom_bodyid[g]).name

    def free(q, data=None, attached=None, strict=False):
        """strict: the attached (held) object must not touch the other objects either."""
        if data is not None:
            probe.qpos[:] = data.qpos
        probe.qpos[kin.qadr] = q
        mujoco.mj_forward(model, probe)  # (a welded object follows the cup through its equality constraint only when
        # simulated: move it with the cup here, keeping its pose relative to the cup as it is in `data`)
        if attached is not None and data is not None:
            _carry(model, data, probe, attached)
        for c in probe.contact[:probe.ncon]:
            if c.dist >= 0:
                continue
            b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
            static1, static2 = b1 not in movable, b2 not in movable
            if static1 and static2:
                hit = True
            elif attached is not None and attached in (b1, b2):
                if c.dist > -HELD_PENETRATION:  # resting on the floor / touching a wall is not a collision
                    continue
                other = c.geom2 if b1 == attached else c.geom1
                if model.geom_bodyid[other] in movable:
                    hit = strict
                else:
                    hit = bool(model.geom_contype[other] & SCENE_BIT and not model.geom_contype[other] & ROBOT_BIT)
            else:
                hit = False
            if hit:
                free.last = (name(c.geom1), name(c.geom2))
                return False
        return True
    free.last = None
    return free


def _carry(model, data, probe, body):
    """Put the free body `body` in `probe` where it sits relative to the cup in `data` (then recompute the contacts)."""
    cup = model.body('cup').id
    rel_pos = data.xmat[cup].reshape(3, 3).T @ (data.xpos[body] - data.xpos[cup])
    rel_quat, inv = np.zeros(4), np.zeros(4)
    mujoco.mju_negQuat(inv, data.xquat[cup])
    mujoco.mju_mulQuat(rel_quat, inv, data.xquat[body])
    pos = probe.xpos[cup] + probe.xmat[cup].reshape(3, 3) @ rel_pos
    quat = np.zeros(4)
    mujoco.mju_mulQuat(quat, probe.xquat[cup], rel_quat)
    adr = model.jnt_qposadr[model.body_jntadr[body]]
    probe.qpos[adr:adr + 3], probe.qpos[adr + 3:adr + 7] = pos, quat
    mujoco.mj_forward(model, probe)
