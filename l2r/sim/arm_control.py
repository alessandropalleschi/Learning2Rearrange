"""Joint-space helpers for a smoother, rate-limited arm controller.

The legacy `Environment.move_joints` sets the position target to `current + 100 * speed * error`, i.e. (for
speed = 0.01) the final goal at once and, for speed = 0.02, twice the error. The PD controller of the simulator then
closes the whole gap in a few steps: large accelerations, large joint reaction forces (which trip the contact-force
stop even in free space) and a `speed` argument that does not mean a speed. The controller here advances a *command*
towards the goal at most `v_max * dt` per simulation step, the way a real robot driver with velocity limits does.
"""
import numpy as np


def rate_limited_command(command, goal, v_max, dt):
    """Move `command` towards `goal` by at most `v_max * dt` per joint (element-wise, arrays)."""
    command, goal, v_max = np.asarray(command, float), np.asarray(goal, float), np.asarray(v_max, float)
    return command + np.clip(goal - command, -v_max * dt, v_max * dt)


def joint_limits(backend, body, joints, prismatic_speed=1.0, prismatic_effort=1000.0):
    """Per-joint (lower, upper, v_max, effort) from the URDF.

    The base slide of this robot is declared with +-10000 m, 10000 m/s and 1 N in the URDF, which are placeholders:
    it gets a workspace-sized range, `prismatic_speed` m/s and `prismatic_effort` N instead."""
    lower, upper, v_max, effort = [], [], [], []
    for j in joints:
        info = backend.getJointInfo(body, j)
        lo, hi, force, vel = info[8], info[9], info[10], info[11]
        if info[2] == backend.JOINT_PRISMATIC:
            lo, hi, vel, force = -0.5, 0.5, prismatic_speed, prismatic_effort
        lower.append(lo)
        upper.append(hi)
        v_max.append(vel if vel > 0 else 3.14)
        effort.append(force)
    return dict(lower=np.array(lower), upper=np.array(upper), v_max=np.array(v_max), effort=np.array(effort))
