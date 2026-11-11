"""Native MuJoCo robot (Menagerie UR5e on the slide, L tool): IK, streaming line moves, a blade push, a suction grasp."""
import os

import numpy as np
import pytest

os.environ.setdefault('MUJOCO_GL', 'egl')
mujoco = pytest.importorskip('mujoco')

from l2r.config import SimParams  # noqa: E402
from l2r.sim.mj import robot  # noqa: E402
from l2r.sim.mj.robot import STAGE_MARGIN  # noqa: E402

if not os.path.exists(os.path.join(robot.UR5E, 'ur5e.xml')):
    pytest.skip('the Menagerie UR5e is not fetched (scripts/fetch_menagerie.sh)', allow_module_level=True)

from l2r.sim.mj.control import Controller, Kinematics  # noqa: E402
from l2r.sim.mj.world import add_box, collision_check, world_spec  # noqa: E402
from l2r.sim.workspace import Workspace  # noqa: E402

HOME = np.array([0, 1, -1, 0.5, 0.5, 0.5, 1]) * np.pi
WS = Workspace.from_params(SimParams())
FLOOR = WS.bounds[2][0]


def horizontal(heading):
    z = np.array([-np.cos(heading), -np.sin(heading), 0.0])
    x = np.array([0, 0, -1.0])
    return np.column_stack([x, np.cross(z, x), z])


def setup(start, rot):
    spec = world_spec(WS)
    add_box(spec, 'block', (0.0, 0.0, FLOOR + 0.051), (0.025, 0.025, 0.05), 0.2)
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    k = Kinematics(m)
    free = collision_check(m)
    q = k.solve_multistart(start, rot, [HOME], np.random.default_rng(0), accept=free)
    assert q is not None
    d.qpos[k.qadr] = q
    mujoco.mj_forward(m, d)
    c = Controller(m, d, k)
    c.command = q.copy()
    c.hold(50)
    c.free = free
    return m, d, c


@pytest.mark.parametrize('heading_deg', [0, 30])
def test_blade_push_moves_the_block_along_the_heading(heading_deg):
    rot = horizontal(np.radians(heading_deg))
    direction = rot[:, 2]
    face = np.array([0.025, 0.0, FLOOR + 0.05])
    start = face - direction * ((WS.bounds[0][1] + STAGE_MARGIN - face[0]) / -direction[0])
    m, d, c = setup(start, rot)
    block = m.body('block').id
    before = d.xpos[block].copy()
    result = c.move_line(face + direction * 0.08, rot, speed=0.1, force_limit=200.0, collision_free=c.free)
    assert result.reached and not result.force_stop
    moved = d.xpos[block] - before
    assert 0.05 < moved[:2] @ direction[:2] < 0.1
    assert d.xmat[block][8] > np.cos(np.radians(5))  # still upright


def test_native_environment_expert_grasp_holds_and_places():
    """The task environment on the native backend: the expert's first grasp welds the object to the cup and carries it out of
    the cabinet; the place puts it back inside, released, with the scene valid."""
    import argparse

    from l2r.config import Config
    from l2r.demos import make_expert
    from l2r.envs.rearrange import make_env
    from l2r.training import add_arguments
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    run = parser.parse_args('--agent object --hierarchy --belief spatial --placement learned --footprint_placement '
                            '--sim_quality modern --grasp_approach clear --backend mujoco'.split())
    env = make_env(Config.from_args(run), False, 5, SimParams.from_args(run), 30000)
    env.reset(num_obj=5, seed=30000)
    choice = make_expert(env).act()
    assert choice[0] == 'grasp'
    _, _, _, _, info = env.step({'primitive': 'grasp', 'pix': choice[1]})
    sim = env.sim
    held = sim.tool.grip_body
    assert info['executed'] and env.holding and sim.holds()
    assert sim.p.getBasePositionAndOrientation(held)[0][0] > env.ws.bounds[0][1]  # out of the cabinet
    col, row = env.obs['heuristic_place']
    _, _, _, _, info = env.step({'primitive': 'place', 'pix': (0, int(row), int(col))})
    assert not sim.holds()
    assert sim.p.getBasePositionAndOrientation(held)[0][0] < env.ws.bounds[0][1]  # back inside
