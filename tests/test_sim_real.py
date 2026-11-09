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
