"""The UR5e arm: loading, joint-space motion, inverse kinematics, straight Cartesian moves and contact detection
from the joint force/torque sensors."""
import logging

import numpy as np

from l2r.sim.arm_control import joint_limits, rate_limited_command
from l2r.sim.force_sensing import ForceMonitor

HOME_JOINTS = np.array([0, 1, -1, 0.5, 0.5, 0.5, 1]) * np.pi  # [base slide, 6 revolute joints]
FT_SENSOR_JOINTS = (11, 10)  # extra joints with a force/torque sensor (besides the tool link)
log = logging.getLogger(__name__)

JOINT_TOLERANCE = 1e-3  # rad
IK_TOLERANCE = 0.002  # m


class Arm:
    """UR5e on a lateral slide, with a pushing blade / suction cup as tool (`ee_link`).

    `holding` and `in_contact` are callbacks set by the environment (state of the suction cup): while an object
    is held the force check starts later, and `in_contact` ends an approach move that touched something.
    """

    def __init__(self, p, params, workspace, time_step):
        self.p, self.params, self.ws, self.time_step = p, params, workspace, time_step
        self.holding = lambda: False
        self.in_contact = lambda: False
        self.force_stops = 0  # motions aborted because the contact force exceeded the limit (diagnostics)
        self.force_events = []  # (tool x, force) of every abort; consumed by profile_sim.py

    # ------------------------------------------------------------------ setup
    def load(self):
        """Load the URDF and look up joints, limits and force sensors. Call after every `resetSimulation`."""
        p = self.p
        self.body = p.loadURDF('assets/ur5e/ur5e.urdf', basePosition=(0.6, 0, 0), useFixedBase=True)
        self.joints = []
        for i in range(p.getNumJoints(self.body)):
            info = p.getJointInfo(self.body, i)
            if info[1].decode('utf-8') == 'fixFrame':
                self.ee_link = info[0]
            if info[2] in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                self.joints.append(info[0])
        sensors = [self.ee_link, *FT_SENSOR_JOINTS]
        for joint in sensors:
            p.enableJointForceTorqueSensor(self.body, joint, 1)
        self.force_monitor = ForceMonitor(p, self.body, sensors, self.params.force_window, self.params.force_debounce)
        self.limits = joint_limits(p, self.body, self.joints)
        self.ik_kwargs = self._ik_kwargs() if self.params.ik_limits else {}

    def _ik_kwargs(self):
        """IK arguments: joint limits from the URDF (base slide limited to the workspace) and the home pose as rest
        pose (null-space pull towards a sensible configuration). PyBullet ignores the URDF limits otherwise."""
        lower, upper = [], []
        for j in self.joints:
            info = self.p.getJointInfo(self.body, j)
            lo, hi = info[8], info[9]
            if info[2] == self.p.JOINT_PRISMATIC:  # base slide: URDF says +-10000 m
                lo, hi = -0.5, 0.5
            lower.append(lo)
            upper.append(hi)
        kwargs = dict(lowerLimits=lower, upperLimits=upper, jointRanges=[u - lo for lo, u in zip(lower, upper)],
                      restPoses=[float(q) for q in HOME_JOINTS[:len(lower)]])
        if self.params.ik_iterations:
            kwargs['maxNumIterations'] = int(self.params.ik_iterations)
        return kwargs

    # ------------------------------------------------------------------ state
    def tool_pose(self):
        """(position, orientation) of the tool link in the world."""
        state = self.p.getLinkState(self.body, self.ee_link)
        return state[4], state[5]

    def tool_frame(self):
        """World frame of the tool link: (position, orientation)."""
        state = self.p.getLinkState(self.body, self.ee_link)
        return state[0], state[1]

    # ------------------------------------------------------------------ joint space
    def go_home(self):
        return self.move_joints(HOME_JOINTS)

    def move_joints(self, target_joints, speed=0.02):
        """Move to a joint configuration. Returns False if it is not reached within the step budget."""
        if self.params.controller == 'pd':
            return self._move_joints_rate_limited(target_joints)
        p = self.p
        for _ in range(self.params.move_joints_max_steps):  # simulation steps, not wall-clock
            current = np.array([s[0] for s in p.getJointStates(self.body, self.joints)])
            diff = target_joints - current
            diff[-1] = 0
            if all(np.abs(diff) < JOINT_TOLERANCE):
                p.stepSimulation()  # give it time to stop
                return True
            # legacy command: current + 100 * speed * error (for speed 0.01 this is the whole goal at once)
            command = current + 100 * diff * speed
            command[-1] = np.pi
            p.setJointMotorControlArray(bodyIndex=self.body, jointIndices=self.joints, controlMode=p.POSITION_CONTROL,
                                        targetPositions=command, positionGains=0.4 * np.ones(len(self.joints)))
            p.stepSimulation()
        return False

    def _move_joints_rate_limited(self, target_joints, tol=JOINT_TOLERANCE):
        """Rate-limited joint move (params.controller == 'pd'): the position command advances towards the goal at
        most v_max * dt per step (URDF velocity limits x params.velocity_scale), starting from the current state, so
        there are no command jumps. Done when the joints are within `tol` and (almost) at rest."""
        p = self.p
        goal = np.asarray(target_joints, dtype=float).copy()
        goal[-1] = np.pi  # the last wrist joint is kept fixed, as in the legacy controller
        v_max = self.limits['v_max'] * self.params.velocity_scale
        command = np.array([s[0] for s in p.getJointStates(self.body, self.joints)])
        extra = {'forces': list(self.limits['effort'])} if self.params.pd_motor_forces == 'urdf' else {}
        for _ in range(self.params.move_joints_max_steps):
            states = p.getJointStates(self.body, self.joints)
            q, qd = np.array([s[0] for s in states]), np.array([s[1] for s in states])
            err = goal - q
            err[-1] = 0
            if np.all(np.abs(err) < tol) and np.all(np.abs(qd) < 0.05):
                p.stepSimulation()
                return True
            command = rate_limited_command(command, goal, v_max, self.time_step)
            p.setJointMotorControlArray(bodyIndex=self.body, jointIndices=self.joints, controlMode=p.POSITION_CONTROL,
                                        targetPositions=command, positionGains=0.4 * np.ones(len(self.joints)), **extra)
            p.stepSimulation()
        return False

    # ------------------------------------------------------------------ Cartesian space
    def move_to_pose(self, pose, speed=0.01):
        """Inverse kinematics + joint move to a tool pose (position, quaternion). Returns success."""
        return self.solve_ik(pose, speed)[1]

    def solve_ik(self, pose, speed):
        """Solve and execute IK until the tool is within IK_TOLERANCE of the target position (at most 5 rounds)."""
        for _ in range(5):
            joints = self.p.calculateInverseKinematics(self.body, self.ee_link, targetPosition=pose[0],
                                                       targetOrientation=pose[1], **self.ik_kwargs)
            if not self.move_joints(joints, speed):
                return np.array(joints, dtype=np.float32), False
            error = np.linalg.norm(np.array(pose[0]) - np.array(self.tool_pose()[0]))
            if error < IK_TOLERANCE:
                break
        return np.array(joints, dtype=np.float32), True

    def waypoints(self, pose0, pose1, monitored=True):
        """IK waypoints of a straight move. Legacy: one every params.step_distance. With params.adaptive_step the
        spacing is 5x larger where no contact has to be detected: outside the cabinet front, and on moves that do
        not monitor the force (retreats)."""
        pose0, pose1 = np.asarray(pose0, dtype=float), np.asarray(pose1, dtype=float)
        length = float(np.linalg.norm(pose1 - pose0))
        direction, step = (pose1 - pose0) / length, self.params.step_distance
        if not self.params.adaptive_step:
            return [pose0 + n * direction * step for n in range(int(np.floor(length / step)))]
        front, points, d = self.ws.bounds[0][1] + 0.02, [], 0.0
        while d < length:
            point = pose0 + d * direction
            points.append(point)
            d += step * (5 if (not monitored or point[0] > front) else 1)
        return points

    def move_straight(self, pose0, pose1, rot, speed=0.01, max_force=None, detect_force=False, detect_contact=False):
        """Move along a straight line through IK waypoints, keeping the orientation `rot`.

        detect_force:   abort (False) when the joint reaction force exceeds `max_force` (default: params limit)
        detect_contact: stop early (True) when the tool touches something within 1 cm of the end point
        """
        max_force = self.params.contact_force_limit if max_force is None else max_force
        if np.linalg.norm(np.float32(pose1) - np.float32(pose0)) == 0:
            return True
        waypoints = self.waypoints(pose0, pose1, monitored=detect_force or detect_contact)
        success = True
        if detect_force and self.params.filtered_force:
            self.force_monitor.tare(step=self.p.stepSimulation)  # zero at the start of the motion
        for n, target in enumerate(waypoints):
            success &= self.move_to_pose((target, rot), speed)
            self.p.stepSimulation()
            if not success:
                return success
            if detect_force and target[0] < self._force_check_limit_x() and self._force_exceeded(max_force):
                self.force_events.append((float(target[0]), float(self._last_force)))
                self.move_to_pose((waypoints[max(n - 1, 0)], rot), speed)  # back off one waypoint
                log.debug('force %.0f N at x=%.3f exceeds the limit %s', self._last_force, target[0], max_force)
                self.force_stops += 1
                return False
            if detect_contact and self.in_contact():
                if np.linalg.norm(np.float32(pose1) - np.float32(target)) < 0.01:
                    return True
        success &= self.move_to_pose((pose1, rot), speed)
        return success

    def _force_check_limit_x(self):
        """The force is only checked where the tool x is below this value."""
        if self.params.force_sense_x is not None:
            return self.params.force_sense_x
        return 0.2 if self.holding() else 0.5

    def _force_exceeded(self, max_force):
        if self.params.filtered_force:
            exceeded = self.force_monitor.update(max_force)
            self._last_force = self.force_monitor.last_force
        else:  # legacy: raw sum of |Fx|+|Fy|+|Fz| of the three monitored joints against the limit
            forces = [np.sum(np.abs(np.array(self.p.getJointState(self.body, j)[2][:3])))
                      for j in self.force_monitor.joints]
            self._last_force = max(forces)
            exceeded = self._last_force > max_force
        return exceeded
