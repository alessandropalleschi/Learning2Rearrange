"""Inverse kinematics and the streaming Cartesian controller of the native MuJoCo robot.

`Kinematics`: damped least squares on a private MjData (never disturbs the simulation), position + orientation of a site,
joint limits, and a pull towards the home posture in the null space (the arm is redundant with the slide: without it the
solutions drift and flip between configurations).

`Controller`: plan, then execute. A straight tip line is first solved as joint waypoints every centimetre (IK warm started
along the line, pulled towards the home posture, optionally collision checked: a line that cannot be followed is refused before
the robot moves); the joint path is then followed continuously with a smooth time profile within the joint velocity limits (no
stop at every waypoint), the servos tracking the command every `substeps` physics steps. A move ends when the tip is at the end
within tolerance, or early when the flange force / torque sensor exceeds a limit (filtered, debounced; tared at the start of the
move), or a caller's `stop` test is true."""
from dataclasses import dataclass

import mujoco
import numpy as np

from l2r.sim.mj.robot import ARMS, arm_of, joint_names

SLIDE_V_MAX = 0.5  # m/s


def pose_error(target_pos, target_rot, pos, rot, rot_weight=1.0):
    """6-vector (position error, rotation error as a world-frame rotation vector * rot_weight)."""
    q_t, q_c, err = np.zeros(4), np.zeros(4), np.zeros(3)
    mujoco.mju_mat2Quat(q_t, np.asarray(target_rot, float).flatten())
    mujoco.mju_mat2Quat(q_c, np.asarray(rot, float).flatten())
    mujoco.mju_subQuat(err, q_t, q_c)  # in the current frame
    return np.concatenate([np.asarray(target_pos) - pos, rot_weight * (np.asarray(rot).reshape(3, 3) @ err)])


class Kinematics:
    def __init__(self, model, site='tip', home=None, rest_gain=0.05):
        """home: the posture of the controlled joints (slide first if present); default: the arm's registered home, else
        the middle of the joint ranges. The posture pull spares the slide (it follows the target sideways) and the last joint
        (it sets the heading)."""
        self.m, self.d = model, mujoco.MjData(model)
        info = ARMS[arm_of(model)]
        self.names = names = list(joint_names(model))
        self.v_max = np.array(([SLIDE_V_MAX] if names[0] == 'slide' else []) + list(info['v_max']))
        self.rest_gain = rest_gain
        self.null_mode = 'limits'  # null-space objective: 'limits' (away from the joint limits; reach 0.94 vs 0.90, mount_study) or 'home'
        self.rest_weights = np.array([0.0 if (n == 'slide' or n == names[-1]) else 1.0 for n in names])
        self.site = model.site(site).id
        self.joints = [model.joint(n).id for n in names]
        self.qadr = np.array([model.jnt_qposadr[j] for j in self.joints])
        self.dofs = np.array([model.jnt_dofadr[j] for j in self.joints])
        self.low = np.array([model.jnt_range[j][0] for j in self.joints])
        self.high = np.array([model.jnt_range[j][1] for j in self.joints])
        if home is None:
            arm_home = info['home'] if info['home'] is not None else 0.5 * (self.low + self.high)[-len(info['joints']):]
            home = ([0.0] if names[0] == 'slide' else []) + list(arm_home)
        self.home = np.asarray(home, float)
        self._jp, self._jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))

    def forward(self, q):
        """(tip position, tip rotation 3x3) of the arm configuration q (7)."""
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        return self.d.site_xpos[self.site].copy(), self.d.site_xmat[self.site].reshape(3, 3).copy()

    def jacobian(self, q):
        self.forward(q)
        mujoco.mj_comPos(self.m, self.d)
        mujoco.mj_jacSite(self.m, self.d, self._jp, self._jr, self.site)
        return np.vstack([self._jp, self._jr])[:, self.dofs]

    def step(self, q, target_pos, target_rot, damping=1e-3, rot_weight=0.3, max_step=0.2):
        """One damped-least-squares step from q towards the target; returns (new q, error norm before the step)."""
        pos, rot = self.forward(q)
        error = pose_error(target_pos, target_rot, pos, rot, rot_weight)
        J = self.jacobian(q)
        J[3:] *= rot_weight
        JJt = J @ J.T + damping * np.eye(6)
        dq = J.T @ np.linalg.solve(JJt, error)
        if self.rest_gain:  # a secondary objective in the null space (the tip does not move)
            null = np.eye(len(q)) - J.T @ np.linalg.solve(JJt, J)
            if self.null_mode == 'limits':  # away from the joint limits (redundant arms: use the extra freedom for that)
                span = np.minimum(self.high - self.low, 2 * np.pi)
                middle = 0.5 * (self.low + self.high)
                secondary = self.rest_gain * self.rest_weights * (middle - q) / (span / 2) ** 2 * 0.25
            else:  # towards the home posture
                secondary = self.rest_gain * self.rest_weights * (self.home - q)
            dq += null @ secondary
        scale = max(1.0, np.abs(dq).max() / max_step)
        return np.clip(q + dq / scale, self.low, self.high), float(np.linalg.norm(error))

    def solve(self, q0, target_pos, target_rot, iterations=500, tol_pos=1e-3, tol_rot=1e-2):
        q = np.array(q0, float)
        for _ in range(iterations):
            pos, rot = self.forward(q)
            error = pose_error(target_pos, target_rot, pos, rot)
            if np.linalg.norm(error[:3]) < tol_pos and np.linalg.norm(error[3:]) < tol_rot:
                return q, True
            q, _ = self.step(q, target_pos, target_rot)
        return q, False

    def solve_multistart(self, target_pos, target_rot, seeds=(), rng=None, starts=16, accept=None):
        """A solution that passes `accept(q)` (e.g. collision free): from the home posture and `seeds` first; random starts
        only if none of those works (then the one closest to home)."""
        for q0 in [self.home] + list(seeds):
            q, ok = self.solve(np.array(q0, float), target_pos, target_rot)
            if ok and (accept is None or accept(q)):
                return q
        rng = rng or np.random.default_rng(0)
        found = []
        for _ in range(starts):
            q0 = rng.uniform(np.maximum(self.low, -np.pi), np.minimum(self.high, np.pi))
            if self.names[0] == 'slide':
                q0[0] = np.clip(target_pos[1], self.low[0], self.high[0])
            q, ok = self.solve(q0, target_pos, target_rot)
            if ok and (accept is None or accept(q)):
                found.append(q)
        return min(found, key=lambda q: np.abs(q - self.home).sum()) if found else None


@dataclass
class MoveResult:
    reached: bool
    force_stop: bool = False
    stopped: bool = False  # the caller's stop test
    steps: int = 0
    max_force: float = 0.0


class Controller:
    def __init__(self, model, data, kinematics, substeps=4, force_window=5, force_debounce=2):
        self.m, self.d, self.k = model, data, kinematics
        self.substeps = substeps
        by_joint = {int(model.actuator_trnid[a][0]): a for a in range(model.nu) if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT}
        self.acts = np.array([by_joint[j] for j in kinematics.joints])
        self.dt = model.opt.timestep * substeps
        self.force_sensor = model.sensor('ft_force')
        self.force_window, self.force_debounce = force_window, force_debounce
        self.command = self.measured()

    def measured(self):
        return self.d.qpos[self.k.qadr].copy()

    def tip(self):
        return self.d.site_xpos[self.k.site].copy(), self.d.site_xmat[self.k.site].reshape(3, 3).copy()

    def flange_force(self):
        a = self.force_sensor.adr[0]
        return self.d.sensordata[a:a + 3].copy()

    def _tick(self):
        self.d.ctrl[self.acts] = self.command
        for _ in range(self.substeps):
            mujoco.mj_step(self.m, self.d)

    def hold(self, ticks=1):
        for _ in range(ticks):
            self._tick()

    def move_joints(self, goal, tol=3e-3, max_time=6.0):
        """Rate-limited joint move from the current command to `goal`; returns MoveResult."""
        goal = np.clip(np.asarray(goal, float), self.k.low, self.k.high)
        self.command = self.measured()
        for tick in range(int(max_time / self.dt)):
            v = self.k.v_max * 0.5 * self.dt
            self.command = self.command + np.clip(goal - self.command, -v, v)
            self._tick()
            if np.allclose(self.command, goal) and np.abs(self.measured() - goal).max() < tol \
                    and np.abs(self.d.qvel[self.k.dofs]).max() < 0.05:
                return MoveResult(True, steps=(tick + 1) * self.substeps)
        return MoveResult(False, steps=int(max_time / self.dt) * self.substeps)

    def plan_line(self, end_pos, end_rot=None, step=0.01, collision_free=None):
        """Joint waypoints of the straight tip line from the current tip to `end_pos` (every `step` m, IK warm started from the
        previous waypoint and pulled towards the home posture), or None if a waypoint has no solution or `collision_free(q)`
        refuses it. The path the robot then follows is the one that was checked."""
        start_pos, start_rot = self.tip()
        end_rot = start_rot if end_rot is None else np.asarray(end_rot)
        n = max(1, int(np.ceil(np.linalg.norm(np.asarray(end_pos) - start_pos) / step)))
        q, path = self.measured(), [self.measured()]
        for i in range(1, n + 1):
            q, ok = self.k.solve(q, start_pos + (np.asarray(end_pos) - start_pos) * i / n, end_rot)
            if not ok or (collision_free is not None and not collision_free(q)):
                return None
            path.append(q)
        return np.array(path)

    def move_line(self, end_pos, end_rot=None, speed=0.1, force_limit=None, stop=None, tol=2e-3, max_time=10.0,
                  collision_free=None, path=None):
        """Tip along a straight line to `end_pos` at about `speed` m/s: the joint path of `plan_line` (or `path`) followed with a
        smooth time profile and the joint velocity limits. Returns MoveResult (not reached and no steps when there is no plan)."""
        path = self.plan_line(end_pos, end_rot, collision_free=collision_free) if path is None else path
        if path is None:
            return MoveResult(False)
        start_pos = self.tip()[0]
        length = float(np.linalg.norm(np.asarray(end_pos) - start_pos))
        duration = max(length / speed, self.dt)
        segments = len(path) - 1
        history, over, peak = [], 0, 0.0
        bias = self.flange_force()
        self.command = self.measured()
        for tick in range(int(max_time / self.dt)):
            s = min(1.0, (tick + 1) * self.dt / duration)
            u = s * s * (3 - 2 * s) * segments  # smooth start and stop, in path segments
            i = min(int(u), segments - 1) if segments else 0
            target = path[i] + (u - i) * (path[min(i + 1, segments)] - path[i]) if segments else path[0]
            v = self.k.v_max * self.dt
            self.command = self.command + np.clip(target - self.command, -v, v)
            self._tick()
            if force_limit is not None:
                history.append(np.linalg.norm(self.flange_force() - bias))
                filtered = float(np.median(history[-self.force_window:]))
                peak = max(peak, filtered)
                over = over + 1 if filtered > force_limit else 0
                if over >= self.force_debounce:
                    return MoveResult(False, force_stop=True, steps=(tick + 1) * self.substeps, max_force=peak)
            if stop is not None and stop():
                return MoveResult(True, stopped=True, steps=(tick + 1) * self.substeps, max_force=peak)
            if s >= 1.0 and np.linalg.norm(self.tip()[0] - end_pos) < tol:
                return MoveResult(True, steps=(tick + 1) * self.substeps, max_force=peak)
        return MoveResult(False, steps=int(max_time / self.dt) * self.substeps, max_force=peak)
