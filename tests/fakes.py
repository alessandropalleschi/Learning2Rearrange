"""Test doubles: a fake simulator with the interface `RearrangeEnv` uses, and fake PyBullet backends."""
from types import SimpleNamespace

import numpy as np

from l2r.learning.object_centric import CLASS_RGB
from l2r.sim.vision import relabel_colormap
from l2r.sim.workspace import Workspace

ROWS, COLS = Workspace().heightmap_shape  # un-padded maps: classmap / idmap / freespace
H, W = ROWS + 8, COLS + 4  # padded maps: colormap / depthmap (4 px border: rows both sides, columns left)


class FakeSim:
    """Stand-in for `l2r.sim.environment.Environment`: sorted after 3 actions, relabelling possible at the 2nd."""

    force_stops = 0

    def __init__(self, gui=False, num_obj=(1,), params=None, client=None):
        self.num_obj, self.t, self.ws = list(num_obj), 0, Workspace()
        self.scene = SimpleNamespace(relabel_ids=[1], relabel_classes=[1], progress=self._progress, positions=self._positions,
                                     regions={'blue': [-0.28, -0.18], 'green': [-0.05, 0.05], 'yellow': [0.18, 0.28]})
        self.moving = True  # objects move a little with every action
        self.force_events = []

    def reset(self):
        self.t = 0
        return True

    def _progress(self):  # (sorted fraction, mean distance to the target region in m): improves with every action
        fraction = min(self.t / 3, 1.0)
        return fraction, 0.2 * (1 - fraction)

    def _positions(self):
        return np.array([[0.0, 0.1 * self.t if self.moving else 0.0, 0.0]])

    def observe(self):
        depth = np.zeros((H, W), np.float32)
        depth[40:60, 50:70] = 0.05
        classes = (depth > 0).astype(np.uint8)[4:-4, 4:]
        color = np.random.randint(0, 254, (H, W, 3), dtype=np.uint8)
        color[:, :30] = 105  # occluded (grey) cells
        color[40:60, 50:70] = CLASS_RGB[0]  # the object is blue (a class colour), so colour-based instance extraction finds it
        return {'colormap': color, 'depthmap': depth, 'classmap': classes, 'idmap': classes.astype(np.float32),
                'freespace': np.ones((ROWS, COLS), np.float32)}

    def target_pixels(self):
        return [10, 50, 90], [30, 70, 110], None

    def push(self, *args, **kwargs):
        return True

    pick = place = push

    def is_valid(self):
        return True

    def fraction_sorted(self):
        return 1.0 if self.t >= 3 else 0.0

    def relabel_candidates(self):
        return 1.0 if self.t == 2 else 0.0

    def relabel_colormap(self, colormap, id_map, ids=None, classes=None):  # the real implementation: padded / un-padded shapes
        ids = self.scene.relabel_ids if ids is None else ids
        classes = self.scene.relabel_classes if classes is None else classes
        return relabel_colormap(colormap, id_map, ids, classes)

    def wait_static(self):
        self.t += 1


class FakeJoints:
    """PyBullet backend with a first-order position controller: q moves a fraction `kp` towards the command."""

    POSITION_CONTROL, JOINT_PRISMATIC, JOINT_REVOLUTE = 0, 1, 0

    def __init__(self, n=7, kp=0.4):
        self.q, self.qd, self.target, self.kp, self.history = np.zeros(n), np.zeros(n), np.zeros(n), kp, []

    def getJointStates(self, body, joints):
        return [(self.q[i], self.qd[i]) for i in range(len(self.q))]

    def setJointMotorControlArray(self, bodyIndex, jointIndices, controlMode, targetPositions, positionGains, **kw):
        self.target = np.array(targetPositions)

    def stepSimulation(self):
        new = self.q + self.kp * (self.target - self.q)
        self.qd, self.q = (new - self.q) * 480, new
        self.history.append(self.q.copy())

    def getJointInfo(self, body, j):
        prismatic = j == 0
        return (j, b'j', 1 if prismatic else 0, 0, 0, 0, 0, 0, -10000.0 if prismatic else -6.28,
                10000.0 if prismatic else 6.28, 1.0 if prismatic else 150.0, 10000.0 if prismatic else 3.14)

    def getLinkState(self, body, link):
        return ((0.5, 0.0, 0.05), (0, 0, 0, 1), (0, 0, 0), (0, 0, 0, 1), (0.5, 0.0, 0.05), (0, 0, 0, 1))

    def getBasePositionAndOrientation(self, body):
        return getattr(self, 'object_position', (0.5, 0.0, 0.0)), (0, 0, 0, 1)


class ScriptedSensors:
    """Joint state returning a scripted sequence of reaction wrenches."""

    def __init__(self, sequence):
        self.sequence, self.k = sequence, 0

    def getJointState(self, body, joint):
        wrench = self.sequence[min(self.k, len(self.sequence) - 1)]
        self.k += 1
        return (0, 0, wrench, 0)
