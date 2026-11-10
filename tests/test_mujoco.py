"""MuJoCo backend (l2r.sim.mj): PyBullet-convention math, same scenes as PyBullet for a seed, the observation pipeline, the
arm, the suction weld, saved states, and the unsupported options."""
import os
import random

import numpy as np
import pytest

os.environ.setdefault('MUJOCO_GL', 'egl')  # before mujoco is imported: headless rendering
mujoco = pytest.importorskip('mujoco')
pybullet = pytest.importorskip('pybullet')
if not hasattr(pybullet, 'connect'):
    pytest.skip('needs the real PyBullet for the comparison', allow_module_level=True)

from l2r.config import SimParams  # noqa: E402
from l2r.sim.mj import transforms as T  # noqa: E402


def params():
    p = SimParams()
    p.controller, p.solver_iterations, p.adaptive_step, p.joint_tolerance = 'pd', 100, True, 3e-3
    return p


def reset(env_class, seed, num_obj=3):
    env = env_class(num_obj=(num_obj,), params=params())
    np.random.seed(seed)
    random.seed(seed)
    assert env.reset()
    return env


@pytest.fixture(scope='module')
def mj_env():
    from l2r.sim.mj.environment import MujocoEnvironment
    return reset(MujocoEnvironment, 5)


def test_transforms_match_pybullet():
    rng = np.random.default_rng(0)
    for _ in range(50):
        euler = rng.uniform(-3, 3, 3)
        q = pybullet.getQuaternionFromEuler(euler)
        np.testing.assert_allclose(T.quat_from_euler(euler), q, atol=1e-9)
        np.testing.assert_allclose(T.euler_from_quat(q), pybullet.getEulerFromQuaternion(q), atol=1e-9)
        np.testing.assert_allclose(T.matrix_from_quat(q), pybullet.getMatrixFromQuaternion(q), atol=1e-9)
        q_back = np.array(T.quat_from_matrix(pybullet.getMatrixFromQuaternion(q)))
        np.testing.assert_allclose(q_back * np.sign(q_back @ q), q, atol=1e-9)
        a, b = rng.normal(size=3), rng.normal(size=3)
        q2 = pybullet.getQuaternionFromEuler(rng.uniform(-3, 3, 3))
        mine, theirs = T.multiply_transforms(a, q, b, q2), pybullet.multiplyTransforms(a, q, b, q2)
        np.testing.assert_allclose(mine[0], theirs[0], atol=1e-5)
        np.testing.assert_allclose(np.array(mine[1]) * np.sign(np.dot(mine[1], theirs[1])), theirs[1], atol=1e-5)


def test_same_scene_and_maps_as_pybullet(mj_env):
    from l2r.sim.environment import Environment
    pb_env = reset(Environment, 5)
    assert dict(pb_env.scene.names) == dict(mj_env.scene.names)
    np.testing.assert_allclose(mj_env.scene.positions()[:, :2], pb_env.scene.positions()[:, :2], atol=0.01)
    a, b = pb_env.observe(), mj_env.observe()
    assert set(a) == set(b)
    for key in a:
        assert np.shape(a[key]) == np.shape(b[key]), key
    assert np.mean(a['classmap'] != b['classmap']) < 0.03
    assert np.mean(np.abs(a['depthmap'] - b['depthmap']) > 0.005) < 0.03


def test_arm_reaches_an_ik_target(mj_env):
    arm = mj_env.arm
    position, orientation = arm.tool_pose()
    target = np.asarray(position) + [-0.05, 0.03, -0.02]
    assert arm.move_to_pose((target, orientation))
    assert np.linalg.norm(np.asarray(arm.tool_pose()[0]) - target) < 0.003
    assert arm.go_home()


def test_expert_grasp_holds_the_object_with_the_suction_weld():
    """The scripted expert's first grasp (front face from the observation) through the task environment: the weld carries the
    object out of the cabinet with the tool."""
    import argparse

    from l2r.config import Config
    from l2r.demos import make_expert
    from l2r.envs.rearrange import make_env
    from l2r.training import add_arguments
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    run = parser.parse_args('--agent object --hierarchy --belief spatial --placement learned --footprint_placement '
                            '--adaptive_step --joint_tolerance 3e-3 --sim_quality modern --backend mujoco'.split())
    env = make_env(Config.from_args(run), False, 3, SimParams.from_args(run), 30000)
    env.reset(num_obj=3, seed=30000)
    choice = make_expert(env).act()
    assert choice[0] == 'grasp'
    target = int(env.obs['idmap'][choice[1][1], choice[1][2]])
    _, _, _, _, info = env.step({'primitive': 'grasp', 'pix': choice[1]})
    assert info['executed'] and env.sim.tool.activated and env.sim.tool.grip_body == target and env.sim.tool.holds()
    assert env.sim.p.getBasePositionAndOrientation(target)[0][0] > env.ws.bounds[0][1]  # out of the cabinet with the tool


def test_saved_state_round_trip(mj_env):
    p = mj_env.p
    before = mj_env.scene.positions().copy()
    state = p.saveState()
    mj_env.push((before[0][0] + 0.06, before[0][1]), angle=0.0, push_length=0.08)
    p.restoreState(stateId=state)
    p.removeState(state)
    np.testing.assert_allclose(mj_env.scene.positions(), before, atol=1e-9)


@pytest.mark.parametrize('field, value', [('pusher', 'paddle'), ('end_effector', 'gripper')])
def test_unported_options_say_so(field, value):
    from l2r.sim.mj.environment import MujocoEnvironment
    p = params()
    setattr(p, field, value)
    with pytest.raises(NotImplementedError):
        MujocoEnvironment(num_obj=(3,), params=p)


def test_backend_flag_selects_the_simulator():
    import argparse

    from l2r.envs.rearrange import make_simulator
    from l2r.sim.environment import Environment
    from l2r.sim.mj.environment import MujocoEnvironment
    parser = argparse.ArgumentParser()
    SimParams.add_arguments(parser)
    assert SimParams.from_args(parser.parse_args([])).backend == 'pybullet'
    chosen = SimParams.from_args(parser.parse_args(['--backend', 'mujoco']))
    assert make_simulator(chosen) is MujocoEnvironment
    assert make_simulator(SimParams()) is Environment
