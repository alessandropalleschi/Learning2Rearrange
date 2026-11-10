"""Return decomposition: credit moves to the step that decides the outcome."""
import numpy as np
import torch

from l2r.learning.rudder import STEP_FEATURES, Rudder


def test_credit_goes_to_the_early_step_that_decides_the_outcome():
    """Episodes of 6 steps; at step 1 a 'clearing' action happens or not (feature 0). The episode succeeds (return 0) only if it
    happened, else it fails (-10) at the end. RUDDER must put the reward difference on step 1, not on the last step."""
    torch.manual_seed(0)
    rng = np.random.RandomState(0)
    rudder = Rudder(lr=3e-3, batch=32, device='cpu')

    def episode(cleared):
        steps = rng.normal(0, 0.1, (6, STEP_FEATURES)).astype(np.float32)
        steps[:, 0] = 0.0
        steps[1, 0] = 1.0 if cleared else -1.0
        return steps, 0.0 if cleared else -10.0
    for _ in range(400):
        rudder.add(*episode(rng.rand() < 0.5))
    for _ in range(400):
        rudder.train(rng=rng)
    good, ret_good = episode(True)
    bad, ret_bad = episode(False)
    r_good, r_bad = rudder.redistribute(good, ret_good), rudder.redistribute(bad, ret_bad)
    assert np.isclose(r_good.sum(), ret_good, atol=1e-4) and np.isclose(r_bad.sum(), ret_bad, atol=1e-4)
    assert r_good[1] - r_bad[1] > 7  # the decisive step carries (most of) the 10-point difference
    assert abs(r_good[-1] - r_bad[-1]) < 3  # not the last step, where the original reward arrives
