"""Real-simulator regression: the scripted scenario must reproduce the recorded results (see sim_scenario.py).

Slow (minutes) and tied to the machine that recorded the golden files: run with  L2R_GOLDEN=1 pytest tests/test_sim_regression.py
"""
import json
import os

import pytest

pytest.importorskip('pybullet')
if not hasattr(pytest.importorskip('pybullet'), 'connect') or not os.environ.get('L2R_GOLDEN'):
    pytest.skip('set L2R_GOLDEN=1 to run the real-simulator regression', allow_module_level=True)

import sim_scenario  # noqa: E402


@pytest.mark.parametrize('quality', ['legacy', 'modern'])
def test_scripted_scenario_matches_recorded_results(quality):
    with open(os.path.join(sim_scenario.GOLDEN, quality + '.json')) as f:
        expected = json.load(f)
    assert sim_scenario.compare(expected, sim_scenario.run(quality)) == []
