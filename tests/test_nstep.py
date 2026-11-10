"""n-step returns for the replay."""
import numpy as np
import pytest

from l2r.learning.memory import Transition
from l2r.learning.nstep import NStepAccumulator


def step(t, reward, done=False, held_next=False):
    return Transition({'t': t}, (0, t, t), {'t': t + 1, **({'held_class': 1} if held_next else {})}, reward, 'push', done)


def test_returns_discounts_and_episode_end():
    acc = NStepAccumulator(3, 0.9)
    assert acc.add(0, step(0, -1.0)) == [] and acc.add(0, step(1, -1.0)) == []
    first = acc.add(0, step(2, -1.0))
    assert len(first) == 1 and first[0].state['t'] == 0 and first[0].next_state['t'] == 3
    assert first[0].reward == pytest.approx(-1 - 0.9 - 0.81) and first[0].state['_nstep_discount'] == pytest.approx(0.9 ** 3)
    end = acc.add(0, step(3, 0.0, done=True))  # the goal: the three pending starts all end there
    assert [t.state['t'] for t in end] == [1, 2, 3] and all(t.done for t in end)
    assert end[0].reward == pytest.approx(-1 - 0.9 + 0.0) and end[-1].reward == 0.0


def test_grasp_to_place_is_not_discounted_and_truncation_bootstraps():
    acc = NStepAccumulator(3, 0.9)
    acc.add(1, step(0, 0.0, held_next=True))  # grasp: the next state holds the object
    acc.add(1, Transition({'t': 1, 'held_class': 1}, (0, 1, 1), {'t': 2}, -1.0, 'place', False))
    cut = acc.flush(1)  # truncated: bootstrap from the last next state
    assert [t.state['t'] for t in cut] == [0, 1] and not any(t.done for t in cut)
    assert cut[0].reward == pytest.approx(0.0 + 1.0 * -1.0) and cut[0].state['_nstep_discount'] == pytest.approx(1.0 * 0.9)
    assert acc.flush(1) == [] and np.isclose(cut[1].state['_nstep_discount'], 0.9)
