"""Contact detection from the joint force/torque sensors of the simulated arm.

PyBullet's `getJointState(...)[2]` returns the joint *reaction* wrench (Fx, Fy, Fz, Mx, My, Mz) in the joint
frame. It includes the weight of everything attached beyond the joint, carries a different static offset in every
arm configuration, and spikes for single simulation steps when contacts are (de)activated. Comparing the raw value
with a fixed threshold therefore gives false stops. `ForceMonitor` does what a real F/T sensor driver does:

  * `tare()` records the wrench at the start of a motion (bias removal, "zeroing"),
  * `update()` subtracts that bias, takes the median over a short window (spike rejection), and
  * only reports a contact after the filtered force stayed above the limit for several consecutive samples.
"""
from collections import deque

import numpy as np


class ForceMonitor:
    def __init__(self, backend, body, joints, window=5, debounce=2):
        """backend: the pybullet module (or a stand-in); body: body id; joints: joint indices with enabled sensors."""
        self.p, self.body, self.joints = backend, body, list(joints)
        self.window, self.debounce = window, debounce
        self.bias = np.zeros((len(self.joints), 6))
        self._history = deque(maxlen=window)
        self._over = 0

    def read(self):
        """Raw reaction wrenches of all monitored joints, shape (n_joints, 6)."""
        return np.array([self.p.getJointState(self.body, j)[2] for j in self.joints], dtype=float)

    def tare(self, samples=3, step=None):
        """Use the current (contact-free) wrench as zero. `step` is called between samples (e.g. stepSimulation)."""
        readings = []
        for _ in range(samples):
            if step is not None:
                step()
            readings.append(self.read())
        self.bias = np.mean(readings, axis=0)
        self.reset_filter()

    def reset_filter(self):
        self._history.clear()
        self._over = 0

    def force(self):
        """Filtered force magnitude (N) per joint after bias removal, shape (n_joints,)."""
        self._history.append(np.linalg.norm((self.read() - self.bias)[:, :3], axis=1))
        return np.median(np.array(self._history), axis=0)

    def update(self, limit):
        """Feed one new sample; True once the filtered force of any joint exceeded `limit` (N) `debounce` times."""
        self._over = self._over + 1 if np.any(self.force() > limit) else 0
        return self._over >= self.debounce
