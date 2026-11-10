"""Validity on the true tilt (opt-in): roll and pitch of 1 rad each is a 73 degree tilt."""
from types import SimpleNamespace

import numpy as np
import pybullet

from l2r.sim.scene import Scene
from l2r.sim.workspace import Workspace


class StubClient:
    """PyBullet's math functions, no physics."""
    getEulerFromQuaternion = staticmethod(pybullet.getEulerFromQuaternion)
    getMatrixFromQuaternion = staticmethod(pybullet.getMatrixFromQuaternion)

    def stepSimulation(self):
        pass


def scene_with(euler, limit):
    scene = Scene.__new__(Scene)
    scene.p, scene.ws, scene.params = StubClient(), Workspace(), SimParamsLike(limit)
    quaternion = pybullet.getQuaternionFromEuler(euler)
    scene.poses = lambda: {7: ((0.0, 0.0, scene.ws.bounds[2][0] + 0.05), quaternion)}
    return scene


def SimParamsLike(limit):
    return SimpleNamespace(tilt_limit_deg=limit)


def test_true_tilt_limit():
    upright = scene_with((0.0, 0.0, 0.3), 69.0)
    assert upright.is_valid() and upright.invalid_objects() == []
    tilted = scene_with((1.0, 1.0, 0.0), 69.0)
    assert np.isclose(tilted.tilt_deg(pybullet.getQuaternionFromEuler((1.0, 1.0, 0.0))), np.degrees(np.arccos(np.cos(1.0) ** 2)))
    assert not tilted.is_valid() and tilted.invalid_objects() == [7]
    legacy = scene_with((1.0, 1.0, 0.0), None)  # the original roll / pitch check lets it pass
    assert legacy.is_valid() and legacy.invalid_objects() == []
