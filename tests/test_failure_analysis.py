"""Summary of the failure analysis: per-episode grasp rates, force stops and tilt counts."""
import importlib.util
import os

import numpy as np

spec = importlib.util.spec_from_file_location('failure_analysis', os.path.join(os.path.dirname(__file__), '..', 'scripts', 'failure_analysis.py'))
fa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fa)


def step(primitive, executed=True, force_stops=0):
    return {'primitive': primitive, 'executed': executed, 'force_stops': force_stops, 'moved': 1, 'target_misplaced': True,
            'misplaced_before': 1, 'misplaced_after': 0, 'invalid': False, 'invalid_objects': [], 'held': None,
            'placed_in_region': True if primitive == 'place' else None, 'q': None}


def test_per_episode_grasp_rate_force_stops_and_tilt():
    episodes = [
        {'outcome': 'sorted', 'reason': '', 'actions': 2, 'max_tilt_deg': 25.0,
         'steps': [step('grasp'), step('place'), step('grasp', False, force_stops=1)]},  # grasp rate 0.5
        {'outcome': 'sorted', 'reason': '', 'actions': 1, 'max_tilt_deg': 3.0, 'steps': [step('grasp'), step('place')]},  # 1.0
        {'outcome': 'out of actions', 'reason': 'looping', 'actions': 4, 'max_tilt_deg': 60.0,
         'steps': [step('grasp', False)] * 4},  # 0.0, many grasps: it would dominate a pooled rate
        {'outcome': 'invalid', 'reason': 'push', 'actions': 1, 'max_tilt_deg': 80.0, 'steps': [step('push')]},  # no grasp
    ]
    s = fa.summarize(episodes)
    assert np.isclose(s['grasp executed'], 2 / 7)  # pooled
    assert np.isclose(s['grasp executed (median per episode)'], 0.5)  # median of 0.5, 1.0, 0.0
    assert np.isclose(s['grasps stopped by the force limit'], 1 / 7)
    assert s['sorted episodes with an object tilted > 20 deg'] == 1
    assert all(np.isfinite(v) for k, v in s.items() if isinstance(v, float) and 'force' in k or 'median' in k)
