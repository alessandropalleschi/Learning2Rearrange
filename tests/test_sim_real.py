"""Smoke tests of the real PyBullet simulation (skipped when PyBullet is not installed)."""
import random

import numpy as np
import pytest

pybullet = pytest.importorskip('pybullet')
if not hasattr(pybullet, 'connect'):
    pytest.skip('PyBullet is stubbed', allow_module_level=True)

from l2r.config import SimParams  # noqa: E402
from l2r.sim.constants import LABELS  # noqa: E402
from l2r.sim.environment import Environment  # noqa: E402


@pytest.fixture(scope='module')
def sim():
    env = Environment(num_obj=[3], params=SimParams())
    np.random.seed(5)
    random.seed(5)
    assert env.reset()
    return env


def test_observation_shapes_and_contents(sim):
    obs = sim.observe()
    rows, cols = sim.ws.heightmap_shape
    assert obs['colormap'].shape == (rows + 8, cols + 4, 3) and obs['depthmap'].shape == (rows + 8, cols + 4)
    for key in ('classmap', 'idmap', 'freespace'):  # not padded
        assert obs[key].shape == (rows, cols)
    assert set(np.unique(obs['classmap'])) <= {0, *LABELS.values()}
    assert len(np.unique(obs['idmap'])) - 1 == 3  # three objects are visible
    assert set(np.unique(obs['freespace'])) <= {0.0, 1.0}
    assert 0 < (obs['depthmap'][4:-4, 4:] > 0.008).mean() < 0.5  # objects cover part of the cabinet floor


def test_task_state_queries(sim):
    assert sim.is_valid()
    assert 0 <= sim.fraction_sorted() <= 1
    assert 0 <= sim.relabel_candidates() <= 1
    low, high, colors = sim.target_pixels()
    assert len(low) == len(high) == len(colors) == 3 and all(lo < hi for lo, hi in zip(low, high))


def test_primitives_execute_and_leave_a_valid_scene(sim):
    obs = sim.observe()
    ys, xs = np.nonzero(obs['classmap'])
    y, x = int(ys[len(ys) // 2]), int(xs[len(xs) // 2])
    assert isinstance(sim.push(sim.ws.px_to_xy(max(x - 4, 0), y), -np.pi / 2), (bool, np.bool_))
    sim.wait_static()
    assert isinstance(sim.pick(sim.ws.px_to_xy(x, y)), (bool, np.bool_))
    sim.suction.release()
    sim.arm.go_home()
    assert sim.is_valid()


def test_learned_perception_path_with_an_oracle(sim):
    """A perception that returns the true masks but is treated as learned: the instances are split in the top view and
    the privileged ground truth (gt_idmap, instance_to_object) is attached; the maps equal the ground-truth ones."""
    from l2r.perception.base import GroundTruthPerception, Segmentation

    class Oracle:
        is_ground_truth = False

        def segment(self, color, depth, object_ids, scene):
            instances = np.where(np.isin(object_ids, list(scene.names)), object_ids, 0)
            return Segmentation(instances, dict(scene.names), {uid: uid for uid in scene.names})

    reference = sim.observe()
    sim.perception = Oracle()
    try:
        learned = sim.observe()
    finally:
        sim.perception = GroundTruthPerception()
    assert (learned['classmap'] == reference['classmap']).all()
    assert (learned['gt_idmap'] == reference['idmap']).all()
    assert all(uid in sim.scene.names for uid in learned['instance_to_object'].values())
    assert set(learned['instance_to_object']) == {int(i) for i in np.unique(learned['idmap']) if i != 0}


def test_parallel_jaw_gripper_picks_and_places_an_object_it_can_reach():
    """Oracle: jaws at the true centre of the front-most object, yawed to its orientation, hold it (>= 4 of 5 scenes), and the
    object can be placed back with the object-in-tool offsets."""
    env = Environment(num_obj=[3], params=SimParams(end_effector='gripper'))
    p, successes = env.p, 0
    for seed in range(1, 6):
        env.reset(seed=seed)
        ids = [i for i in env.scene.object_ids if abs(p.getBasePositionAndOrientation(i)[0][1]) < 0.17]
        uid = max(ids, key=lambda i: p.getBasePositionAndOrientation(i)[0][0])
        pos, orn = p.getBasePositionAndOrientation(uid)
        theta = p.getEulerFromQuaternion(orn)[2]
        wrap = lambda a: (a + np.pi / 2) % np.pi - np.pi / 2  # jaws are symmetric under 180 degrees
        candidates = [a for a in (wrap(theta), wrap(theta + np.pi / 2)) if abs(a) <= np.deg2rad(60)] or [0.0]
        if env.pick(pos[:2], yaw=candidates[0]):
            successes += 1
            if seed == 1:
                assert env.tool.holds() or not env.params.suction_verify
                assert env.place((0.0, 0.0)) in (True, False) and not env.tool.activated
    assert successes >= 4
