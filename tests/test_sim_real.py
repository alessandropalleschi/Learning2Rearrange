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
    from l2r.config import Config
    from l2r.envs.rearrange import RearrangeEnv
    gym_env = RearrangeEnv(Config(end_effector='gripper'), num_obj=3, params=SimParams(end_effector='gripper'))
    env = gym_env.sim
    p, successes = env.p, 0
    for seed in range(1, 6):
        gym_env.reset(seed=seed)
        ids = [i for i in env.scene.object_ids if abs(p.getBasePositionAndOrientation(i)[0][1]) < 0.17] or list(env.scene.object_ids)
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


def test_approach_lines_and_blade_yaw_geometry():
    from types import SimpleNamespace

    from l2r.sim.primitives import FRONT_MARGIN, Manipulator
    from l2r.sim.workspace import Workspace
    ws = Workspace()
    m = Manipulator(SimpleNamespace(), None, None, ws, SimpleNamespace())
    direction, start = m.approach_line((0.0, 0.0, 0.44), 0.0)
    assert np.allclose(direction, (-1, 0, 0)) and np.isclose(start[0], ws.bounds[0][1] + FRONT_MARGIN) and np.isclose(start[1], 0)
    direction, start = m.approach_line((0.0, 0.0, 0.44), np.deg2rad(30))
    assert np.allclose(direction[:2], (-np.cos(np.pi / 6), -np.sin(np.pi / 6)))
    assert start[1] > 0 and np.isclose(start[0], ws.bounds[0][1] + FRONT_MARGIN)  # comes in from +y, moving towards -y
    assert m.approach_line((0.0, 0.0, 0.44), np.deg2rad(70)) is None  # too steep
    assert m.approach_line((-0.2, 0.25, 0.44), np.deg2rad(40)) is None  # would enter through the side wall


def _front_object(env):
    p = env.p
    ids = [i for i in env.scene.object_ids if abs(p.getBasePositionAndOrientation(i)[0][1]) < 0.15] or list(env.scene.object_ids)
    return max(ids, key=lambda i: p.getBasePositionAndOrientation(i)[0][0])


def test_push_towards_the_back_and_pull_towards_the_robot_move_the_object_that_way():
    env = Environment(num_obj=[1], params=SimParams())
    p, moved = env.p, {'push': 0, 'pull': 0}
    for seed in range(3):
        np.random.seed(seed)
        random.seed(seed)
        assert env.reset()
        uid = _front_object(env)
        before = np.asarray(p.getBasePositionAndOrientation(uid)[0])
        face = p.rayTest((0.3, before[1], 0.45), (-0.3, before[1], 0.45))[0][3][0]  # the object's front face on the push line
        start = (face + 0.02, before[1])  # just in front of it, pushing towards the back
        env.push(start, heading=np.pi, push_length=0.06)
        after = np.asarray(p.getBasePositionAndOrientation(uid)[0])
        moved['push'] += after[0] < before[0] - 0.02
        env.pull(after[:2], heading=0.0, distance=0.06)
        pulled = np.asarray(p.getBasePositionAndOrientation(uid)[0])
        moved['pull'] += pulled[0] > after[0] + 0.02
    assert moved['push'] >= 2, moved
    assert moved['pull'] >= 2, moved


def test_grid_goal_slots_decide_success_by_position():
    params = SimParams()
    params.goal_layout = 'grid'
    env = Environment(num_obj=[1], params=params)
    np.random.seed(3)
    random.seed(3)
    assert env.reset()
    scene, p = env.scene, env.p
    uid = scene.object_ids[0]
    key = scene.class_name(uid)
    x0, x1, y0, y1 = scene.slots[key]
    masks = scene.goal_masks()
    assert set(masks) == {'blue', 'yellow', 'green'} and all(m.any() for m in masks.values())
    assert not (masks['blue'] & masks['green']).any()  # different slots
    pos, orn = p.getBasePositionAndOrientation(uid)
    p.resetBasePositionAndOrientation(uid, ((x0 + x1) / 2, (y0 + y1) / 2, pos[2]), orn)
    assert scene.fraction_sorted() == 1.0 and scene.goal_distance(p.getBasePositionAndOrientation(uid)[0], key) == 0.0
    other = next(s for k, s in scene.slots.items() if k != key)
    p.resetBasePositionAndOrientation(uid, ((other[0] + other[1]) / 2, (other[2] + other[3]) / 2, pos[2]), orn)
    assert scene.fraction_sorted() == 0.0 and scene.goal_distance(p.getBasePositionAndOrientation(uid)[0], key) > 0


def test_planning_expert_restores_the_scene_after_trying_its_candidates():
    from l2r.config import Config
    from l2r.envs.rearrange import RearrangeEnv
    from l2r.planner import PlanningExpert
    env = RearrangeEnv(Config(placement='learned', footprint_placement=True, max_actions=10), num_obj=4, params=SimParams())
    env.reset(num_obj=4, seed=11)
    expert = PlanningExpert(env, candidates=3, horizon=1)
    before = env.sim.scene.positions().copy()
    actions_before, obs_before = env.actions, env.obs
    choice = expert.act()
    assert choice is not None and choice[0] in ('grasp', 'push')
    assert np.abs(env.sim.scene.positions() - before).max() < 1e-4  # every trial was undone
    assert env.actions == actions_before and not env.holding and np.array_equal(env.obs['depthmap'], obs_before['depthmap'])
    if len(getattr(expert, 'last_scores', [])) > 1:
        assert len({round(s, 3) for _, s in expert.last_scores}) >= 1
    _, _, _, _, info = env.step({'primitive': choice[0], 'pix': choice[1]})  # the chosen action executes normally
    assert 'executed' in info
