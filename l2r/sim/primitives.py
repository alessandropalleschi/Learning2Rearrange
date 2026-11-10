"""The motion primitives: push with the blade, pick with the end effector (suction cup or parallel jaws), place the held
object, and pull (attach with the end effector, drag along the floor, release).

Frames: x is the cabinet depth (the robot stands at +x, outside the opening), y the cabinet width, z up. Parameterised variants
(all opt-in; the defaults are the original single-axis motions):
* push `heading` phi: the pusher's tip (a small round head on a stick that enters straight through the opening) moves along
  (cos phi, sin phi) in the floor plane (pi/2: +y, -pi/2: -y, pi: towards the back); headings with a component towards the robot
  are not pushes (use `pull`). Reach limits the depth of pushes towards the back near the side walls.
* pick `approach` psi: the tool comes in along (-cos psi, -sin psi), i.e. the default straight-in approach rotated by psi about the
  vertical, starting from the cabinet front on that line (infeasible if that start is outside the opening).
* `pull`: approach like a pick, attach (vacuum on / jaws closed), drag by `distance` along `heading` (default 0: towards the
  robot), release and leave along the approach line."""
import logging

import numpy as np

from l2r.sim.arm import HOME_JOINTS

log = logging.getLogger(__name__)

FRONT_MARGIN = 0.1  # m in front of the cabinet where every primitive starts / ends (place: 0.15)
PUSH_HEIGHT = 0.05  # m above the cabinet floor: low, to avoid toppling objects (paper: 5 cm)
PICK_HEIGHT = 0.04
GRASP_HEIGHT = PICK_HEIGHT  # jaw centre above bounds[2][0] (the cabinet floor): pads at mid-height of the objects
PUSH_QUATERNION = (np.sqrt(0.5), 0, -np.sqrt(0.5), 0)  # blade pointing into the cabinet
PLACE_QUATERNION = (0.7088025212287903, 0.0003121407644357532, -0.7054068446159363, 0.0003104391216766089)
WALL_MARGIN = 0.008  # m: a push stops this far from the side walls
# paddle pusher (`--pusher paddle`): a T-shaped plate fixed to the tool during pushes (a tool change / retractable paddle), aligned
# with the cabinet: a side plate (8 cm along the depth, 4 cm tall) pushes along the width, a front plate (6 cm wide) towards the back.
PADDLE_PLATES = (((0.04, 0.005, 0.02), (0.03, 0.0, 0.0)),      # (half extents, centre offset from the tool tip), world axes
                 ((0.005, 0.03, 0.02), (-0.005, 0.0, 0.0)))
PADDLE_MASS = 0.05
OPENING_MARGIN = 0.03  # m: an approach line must enter the cabinet at least this far from the side walls
MAX_APPROACH = np.deg2rad(60)  # rad: steeper approaches are refused (the arm would have to reach around the side walls)


class Manipulator:
    """Push / pick / place built from the arm's Cartesian moves and the suction cup."""

    def __init__(self, arm, tool, scene, workspace, params):
        """tool: the end effector, a `SuctionCup` or a `ParallelJawGripper`."""
        self.arm, self.suction, self.scene, self.ws, self.params = arm, tool, scene, workspace, params

    # ------------------------------------------------------------------ helpers
    def _front(self, y, z, margin=FRONT_MARGIN):
        """Staging point in front of the cabinet (outside it, at the given y / z)."""
        return (self.ws.bounds[0][1] + margin, y, z)

    def _yawed(self, quat, angle):
        """`quat` rotated by `angle` (rad) about the world vertical."""
        p = self.arm.p
        return p.multiplyTransforms([0, 0, 0], p.getQuaternionFromEuler([0, 0, angle]), [0, 0, 0], quat)[1]

    def approach_line(self, target, approach):
        """(unit direction into the cabinet, staging point in front of the cabinet on the line through `target`), or None if the
        line does not enter through the opening (or is steeper than MAX_APPROACH)."""
        if abs(approach) > MAX_APPROACH:
            return None
        direction = np.array([-np.cos(approach), -np.sin(approach), 0.0])
        b = self.ws.bounds
        distance = (b[0][1] + FRONT_MARGIN - target[0]) / np.cos(approach)
        start = np.asarray(target, float) - direction * distance
        entry = np.asarray(target, float) - direction * (b[0][1] - target[0]) / np.cos(approach)  # where it crosses the front
        if not (b[1][0] + OPENING_MARGIN <= entry[1] <= b[1][1] - OPENING_MARGIN):
            return None
        return direction, tuple(start)

    def _attach_paddle(self, front_plate=False):
        """Fix the paddle to the tool where it is now (the tool must already have its push orientation); returns (body, constraint).
        The front plate (pushes towards the back) is only fitted when needed: going in, it sweeps a 6 cm wide corridor."""
        p, arm = self.arm.p, self.arm
        tip, _ = arm.tool_pose()
        plates = PADDLE_PLATES if front_plate else PADDLE_PLATES[:1]
        shapes = dict(shapeTypes=[p.GEOM_BOX] * len(plates), halfExtents=[h for h, _ in plates],
                      collisionFramePositions=[o for _, o in plates])
        collision = p.createCollisionShapeArray(**shapes)
        visual = p.createVisualShapeArray(shapeTypes=shapes['shapeTypes'], halfExtents=shapes['halfExtents'],
                                          visualFramePositions=shapes['collisionFramePositions'], rgbaColors=[[0.3, 0.3, 0.35, 1]] * len(plates))
        body = p.createMultiBody(PADDLE_MASS, collision, visual, basePosition=tip, baseOrientation=(0, 0, 0, 1))
        for link in range(-1, p.getNumJoints(arm.body)):
            p.setCollisionFilterPair(arm.body, body, link, -1, 0)
        frame_pos, frame_orn = arm.tool_frame()
        inv_pos, inv_orn = p.invertTransform(frame_pos, frame_orn)
        rel_pos, rel_orn = p.multiplyTransforms(inv_pos, inv_orn, tip, (0, 0, 0, 1))
        constraint = p.createConstraint(arm.body, arm.ee_link, body, -1, p.JOINT_FIXED, (0, 0, 0), rel_pos, (0, 0, 0),
                                        parentFrameOrientation=rel_orn)
        p.changeConstraint(constraint, maxForce=1000)
        return body, constraint

    def _detach_paddle(self, paddle):
        if paddle is not None:
            self.arm.p.removeConstraint(paddle[1])
            self.arm.p.removeBody(paddle[0])

    def _retreat_from(self, pos, front, quat, speed):
        """Leave towards the staging point on a straight line from `pos`, then go home."""
        self.arm.move_straight(pos, front, quat, speed)
        self.arm.go_home()

    # ------------------------------------------------------------------ push
    def push(self, start_xy, angle=0.0, speed=0.01, push_length=0.08, heading=None):
        """Push along y from `start_xy` by `push_length` m (direction sin(angle) = -1 / +1), or along any floor direction
        `heading` (see the module doc). True if the motion completed and the scene is still valid."""
        if heading is not None:
            return self._push_heading(start_xy, heading, speed, push_length)
        arm, b = self.arm, self.ws.bounds
        arm.move_joints(HOME_JOINTS)
        start = np.array([start_xy[0], start_xy[1], b[2][0] + PUSH_HEIGHT])
        end = [start[0], start[1] + push_length * np.sin(angle), start[2]]
        end[1] = max(min(end[1], b[1, 1] - WALL_MARGIN), b[1, 0] + WALL_MARGIN)
        quat = PUSH_QUATERNION
        front = self._front(start[1], start[2])

        success = arm.move_to_pose((front, quat), speed)
        paddle = self._attach_paddle() if (success and getattr(self.params, 'pusher', 'tip') == 'paddle') else None
        try:
            return self._push_axis(start, end, front, quat, speed, success)
        finally:
            self._detach_paddle(paddle)

    def _push_axis(self, start, end, front, quat, speed, success):
        arm = self.arm
        if success:
            success = arm.move_straight(front, start, quat, speed, detect_force=True)  # approach
        else:
            pos = np.asarray(arm.tool_pose()[0])
            self._retreat_from(pos, self._front(pos[1], pos[2]), quat, speed)
            return success
        if success:
            success = arm.move_straight(start, end, quat, 0.1 * speed, detect_force=True)  # the push itself
        else:
            log.debug('push: approach stopped, going back home')
            pos = np.asarray(arm.tool_pose()[0])
            self._retreat_from(pos, self._front(pos[1], pos[2]), quat, speed)
            return success
        pos = np.asarray(arm.tool_pose()[0])
        arm.move_to_pose((start, quat), speed)
        # retreat: after a completed push from the start point, after a force stop from where it stopped
        self._retreat_from(start if success else pos, self._front(start[1], start[2]), quat, speed)
        return success & self.scene.is_valid()

    def _push_heading(self, start_xy, heading, speed, push_length):
        arm, b = self.arm, self.ws.bounds
        direction = np.array([np.cos(heading), np.sin(heading), 0.0])
        if direction[0] > 1e-6:
            raise ValueError('a push cannot move towards the robot (heading %.2f rad): use pull' % heading)
        arm.move_joints(HOME_JOINTS)
        start = np.array([start_xy[0], start_xy[1], b[2][0] + PUSH_HEIGHT])
        end = start + push_length * direction
        end[0] = max(end[0], b[0][0] + WALL_MARGIN)
        end[1] = max(min(end[1], b[1, 1] - WALL_MARGIN), b[1, 0] + WALL_MARGIN)
        front, quat = self._front(start[1], start[2]), PUSH_QUATERNION  # the stick stays straight in
        reached = arm.move_to_pose((front, quat), speed)
        backwards = np.cos(heading) < -np.cos(np.radians(45))  # within 45 degrees of straight to the back
        paddle = self._attach_paddle(front_plate=backwards) if (reached and getattr(self.params, 'pusher', 'tip') == 'paddle') else None
        try:
            success = reached and arm.move_straight(front, start, quat, speed, detect_force=True)
            if not success:
                pos = np.asarray(arm.tool_pose()[0])
                self._retreat_from(pos, self._front(pos[1], pos[2]), quat, speed)
                return False
            success = arm.move_straight(start, end, quat, 0.1 * speed, detect_force=True)
            pos = np.asarray(arm.tool_pose()[0])
            self._retreat_from(pos, self._front(pos[1], pos[2]), quat, speed)
            return success & self.scene.is_valid()
        finally:
            self._detach_paddle(paddle)

    # ------------------------------------------------------------------ pick
    def pick(self, xy, speed=0.01, yaw=0.0, approach=0.0, height=None):
        """Pick the object at `xy` with the end effector: `_pick_with_cup`, or with the jaws (rotated by `yaw` about the
        vertical) `_pick_with_jaws`. `approach` (rad) rotates the approach line about the vertical (see the module doc);
        `height` (m above the cabinet floor) replaces the fixed contact height (PICK_HEIGHT / GRASP_HEIGHT)."""
        if approach or height is not None:
            return self._pick_along(xy, speed, approach, lift=True, height=height)
        if getattr(self.suction, 'is_gripper', False):
            return self._pick_with_jaws(xy, speed, yaw)
        return self._pick_with_cup(xy, speed)

    def pull(self, xy, approach=0.0, heading=0.0, distance=0.08, speed=0.01, height=None):
        """Attach to the object at `xy` along the approach line, drag it by `distance` m along `heading` on the floor, release, leave.
        True if the object was attached, the drag completed and the scene is still valid."""
        return self._pick_along(xy, speed, approach, lift=False, drag=(heading, distance), height=height)

    def _attach_along(self, xy, speed, approach, height=None):
        """Go in along the approach line to the object at `xy` and attach (cup: contact + vacuum; jaws: close around it).
        Returns (attached, tool position, orientation, staging point), or None if the approach line is infeasible."""
        arm, tool, b = self.arm, self.suction, self.ws.bounds
        gripper = getattr(tool, 'is_gripper', False)
        arm.move_joints(HOME_JOINTS)
        home_quat = arm.tool_pose()[1]
        height = b[2][0] + (height if height is not None else (GRASP_HEIGHT if gripper else PICK_HEIGHT))
        target = np.array([xy[0], xy[1], height])
        line = self.approach_line(target, approach)
        if line is None:
            return None
        direction, front = line
        if gripper:
            target = target - direction * tool.grip_depth  # the middle of the pads goes to the object centre
        quat = self._yawed(home_quat, approach)
        tool.release()
        if hasattr(tool, 'axis'):
            tool.axis = direction  # the cup seals on surfaces facing it along this axis
        reached = arm.move_to_pose((front, quat), speed)
        if not reached:
            return False, np.asarray(arm.tool_pose()[0]), quat, front
        if gripper:
            arm.move_straight(front, tuple(target), quat, 0.5 * speed, detect_force=True)
            attached = tool.close()
        else:
            arm.move_straight(front, tuple(target), quat, 0.5 * speed, detect_contact=True, detect_force=True)
            pos = np.asarray(arm.tool_pose()[0])
            if np.linalg.norm(target - pos) < 0.05:
                tool.activate()
            attached = tool.activated
        return attached, np.asarray(arm.tool_pose()[0]), quat, front

    def _pick_along(self, xy, speed, approach, lift=True, drag=None, height=None):
        arm, tool, b = self.arm, self.suction, self.ws.bounds
        result = self._attach_along(xy, speed, approach, height)
        if result is None:
            return False  # infeasible approach line: nothing moved
        attached, pos, quat, front = result
        success = bool(attached)
        if success and drag is not None:  # pull: slide the object on the floor, then let go
            heading, distance = drag
            end = pos + distance * np.array([np.cos(heading), np.sin(heading), 0.0])
            end[0] = min(end[0], b[0][1] - WALL_MARGIN)
            end[1] = max(min(end[1], b[1, 1] - WALL_MARGIN), b[1, 0] + WALL_MARGIN)
            success = arm.move_straight(tuple(pos), tuple(end), quat, 0.3 * speed, detect_force=True)
            tool.release()
            pos = np.asarray(arm.tool_pose()[0])
        elif success:  # pick: carry it out along the approach line
            leave = np.asarray(front) + (np.asarray(front) - pos) / max(np.linalg.norm(np.asarray(front) - pos), 1e-6) * 0.2
            arm.move_straight(tuple(pos), tuple(leave), quat, speed)
            arm.go_home()
            if self.params.suction_verify and not tool.holds():
                tool.release()
                success = False
            if hasattr(tool, 'axis'):
                tool.axis = None
            held = (tool.held,) if getattr(tool, 'is_gripper', False) and success else ()
            return success & self.scene.is_valid(ignore=held)
        arm.move_straight(tuple(pos), tuple(front), quat, speed)
        arm.go_home()
        if hasattr(tool, 'axis'):
            tool.axis = None
        return success & self.scene.is_valid()

    def _pick_with_cup(self, xy, speed=0.01):
        """Touch the object at `xy` from the front with the cup, switch the vacuum on and lift it out of the
        cabinet. True if an object is held (and, with params.suction_verify, still held after the lift)."""
        arm, suction, b = self.arm, self.suction, self.ws.bounds
        arm.move_joints(HOME_JOINTS)
        target = np.array([xy[0], xy[1], b[2][0] + PICK_HEIGHT])
        pos, quat = arm.tool_pose()
        suction.release()
        front = self._front(target[1], target[2])

        success = arm.move_to_pose((front, quat), speed)
        if not success:
            arm.move_straight(pos, front, quat, speed)
            arm.go_home()
            return success
        success = arm.move_straight(front, target, quat, speed=0.5 * speed, detect_contact=True, detect_force=True)
        pos = np.asarray(arm.tool_pose()[0])
        if np.linalg.norm(np.float32(target) - np.float32(pos)) < 0.05:  # touching the object (or at the target)
            suction.activate()
            success = True
        if success:
            suction.activate()
            lift = self._front(pos[1], pos[2], margin=FRONT_MARGIN + 0.2)
            arm.move_straight(pos, lift, quat, speed)
            arm.go_home()
            grasped = suction.activated
            if grasped and self.params.suction_verify and not suction.holds():
                suction.release()  # the object slipped off during the lift
                grasped = False
            success &= grasped
        else:
            pos = np.asarray(arm.tool_pose()[0])
            arm.move_straight(pos, front, quat, speed)
            arm.go_home()
        arm.go_home()
        return success & self.scene.is_valid()

    def _pick_with_jaws(self, xy, speed=0.01, yaw=0.0):
        """Approach from the front with open jaws rotated by `yaw` about the vertical (fingertips at `xy`, low above the
        floor), close, pull the object out of the cabinet. True if an object is held after the lift (and still between the
        pads with params.suction_verify)."""
        arm, gripper, b = self.arm, self.suction, self.ws.bounds
        arm.move_joints(HOME_JOINTS)
        pos, home_quat = arm.tool_pose()
        quat = self.arm.p.multiplyTransforms([0, 0, 0], self.arm.p.getQuaternionFromEuler([0, 0, yaw]), [0, 0, 0], home_quat)[1]
        height = b[2][0] + GRASP_HEIGHT
        front = self._front(xy[1], height)
        target = (xy[0] + gripper.grip_depth, xy[1], height)  # tool origin: the middle of the pads goes to the object centre
        success = arm.move_to_pose((front, quat), speed)
        if success:
            success = arm.move_straight(front, target, quat, 0.5 * speed, detect_force=True)
        grasped = bool(success) and gripper.close()
        pos = np.asarray(arm.tool_pose()[0])
        arm.move_straight(pos, self._front(pos[1], pos[2], margin=FRONT_MARGIN + 0.2), quat, speed)
        arm.go_home()
        if grasped and self.params.suction_verify and not gripper.holds():
            gripper.release()
            grasped = False
        arm.go_home()
        return grasped & self.scene.is_valid(ignore=(gripper.held,) if grasped else ())

    # ------------------------------------------------------------------ place
    def place(self, xy, speed=0.01):
        """Carry the held object to `xy` (the cup position that puts the object there), release and retreat."""
        arm, suction, b = self.arm, self.suction, self.ws.bounds
        quat = PLACE_QUATERNION
        offset_x, offset_y = suction.place_offsets(quat)
        position = np.array([xy[0] - offset_x, xy[1] + offset_y, b[2][0] + PUSH_HEIGHT])
        front = self._front(position[1], position[2], margin=0.15)

        reached = arm.move_to_pose((front, quat), speed)
        if not reached:
            reached = arm.move_to_pose((front, quat), speed)  # retry once
            reached = reached or not self.params.strict_place  # legacy: carry on even if the retry failed
        placed = reached and arm.move_straight(front, position, quat, speed, detect_force=True)
        suction.release()  # drop the object where the tool is (also after a failed approach)
        pos = np.asarray(arm.tool_pose()[0])
        retreated = arm.move_straight(pos, front, quat, speed)
        arm.go_home()
        arm.go_home()
        valid = self.scene.is_valid()
        suction.release()
        return bool(placed and retreated and valid)
