"""Native MuJoCo robot (Menagerie UR5e on the slide, L tool): IK, streaming line moves, a blade push, a suction grasp."""
import os

import numpy as np
import pytest

os.environ.setdefault('MUJOCO_GL', 'egl')
mujoco = pytest.importorskip('mujoco')

from l2r.config import SimParams  # noqa: E402
from l2r.sim.mj import robot  # noqa: E402

if not os.path.exists(os.path.join(robot.UR5E, 'ur5e.xml')):
    pytest.skip('the Menagerie UR5e is not fetched (scripts/fetch_menagerie.sh)', allow_module_level=True)

from l2r.sim.mj.control import Controller, Kinematics  # noqa: E402
from l2r.sim.mj.world import add_box, world_spec  # noqa: E402
from l2r.sim.workspace import Workspace  # noqa: E402

HOME = np.array([0, 1, -1, 0.5, 0.5, 0.5, 1]) * np.pi
# The mount of the arm for the L tool is undecided (scripts/mount_study.py): at the default base (0.6, 0, 0) the blade meets the
# forearm in these motions now that the tool collides with the arm.
MOUNT_UNDECIDED = pytest.mark.xfail(reason='L-tool mount undecided: the blade meets the forearm at the default base',
                                    strict=False)
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

    def free(q):
        probe = mujoco.MjData(m)
        probe.qpos[k.qadr] = q
        mujoco.mj_forward(m, probe)
        return not any(c.dist < 0 for c in probe.contact[:probe.ncon])
    q = k.solve_multistart(start, rot, [HOME], np.random.default_rng(0), accept=free)
    assert q is not None
    d.qpos[k.qadr] = q
    mujoco.mj_forward(m, d)
    c = Controller(m, d, k)
    c.command = q.copy()
    c.hold(50)
    return m, d, c


@MOUNT_UNDECIDED
@pytest.mark.parametrize('heading_deg', [0, 30])
def test_blade_push_moves_the_block_along_the_heading(heading_deg):
    rot = horizontal(np.radians(heading_deg))
    direction = rot[:, 2]
    face = np.array([0.025, 0.0, FLOOR + 0.05])
    start = face - direction * ((WS.bounds[0][1] + 0.1 - face[0]) / -direction[0])
    m, d, c = setup(start, rot)
    block = m.body('block').id
    before = d.xpos[block].copy()
    result = c.move_line(face + direction * 0.08, rot, speed=0.1, force_limit=200.0)
    assert result.reached and not result.force_stop
    moved = d.xpos[block] - before
    assert 0.05 < moved[:2] @ direction[:2] < 0.1
    assert d.xmat[block][8] > np.cos(np.radians(5))  # still upright


@MOUNT_UNDECIDED
def test_suction_carries_the_block_out_of_the_cabinet():
    rot = horizontal(0.0)
    face = np.array([0.025, 0.0, FLOOR + 0.04])
    start = face + np.array([WS.bounds[0][1] + 0.1 - face[0], 0, 0])
    m, d, c = setup(start, rot)
    assert c.move_line(face - np.array([0.01, 0, 0]), rot, speed=0.05).reached
    d.ctrl[m.actuator('suction').id] = 1.0
    c.hold(20)
    c.move_line(c.tip()[0] + np.array([0, 0, 0.02]), rot, speed=0.05)
    c.move_line(start + np.array([0, 0, 0.02]), rot, speed=0.15)
    assert d.xpos[m.body('block').id][0] > WS.bounds[0][1]
    d.ctrl[m.actuator('suction').id] = 0.0
    c.hold(100)
    assert d.xpos[m.body('block').id][0] > WS.bounds[0][1] - 0.05  # released out there (it falls)
