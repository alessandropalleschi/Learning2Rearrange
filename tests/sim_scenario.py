"""Deterministic scripted scenario on the REAL simulator, used as a regression oracle for refactoring.

    python tests/sim_scenario.py record   # write tests/golden/<quality>.json with the current code
    python tests/sim_scenario.py check    # compare the current code with the recorded results

The scene (object types, sizes, positions, target regions) is seeded and the actions are scripted from the
observation, so the same code gives the same numbers on the same machine. Results are compared with tolerances;
bit-exact equality across different machines / PyBullet builds is not expected (chaotic contact dynamics).
"""
import json
import os
import sys
from types import SimpleNamespace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
GOLDEN = os.path.join(ROOT, 'tests', 'golden')


def _imports():
    from l2r.config import Config, SimParams
    from l2r.envs.rearrange import RearrangeEnv
    return Config, SimParams, RearrangeEnv


def scripted_action(classmap, i):
    """Actions derived from the observation only: grasp / push / grasp / push (other direction)."""
    ys, xs = np.nonzero(classmap)
    if len(ys) == 0:
        return {'primitive': 'push', 'pix': (0, 40, 40)}
    k = len(ys) // 2
    y, x = int(ys[k]), int(xs[k])
    if i % 2 == 0:
        return {'primitive': 'grasp', 'pix': (0, y, x)}
    return {'primitive': 'push', 'pix': (i // 2 % 2, y, max(x - 4, 0))}


def run(quality='legacy', seed=3, steps=4):
    Config, SimParams, RearrangeEnv = _imports()
    args = SimpleNamespace(physics='legacy', randomize=False, step_distance=0.002, move_budget=1000,
                           sim_quality=quality, camera_scale=1.0, force_limit=300.0, force_sense_x=None,
                           adaptive_step=False, velocity_scale=1.0)
    params = SimParams.from_args(args)
    env = RearrangeEnv(Config(max_actions=50), num_obj=3, params=params)
    obs = env.reset(seed=seed)
    log = [{'reset_depth': round(float(obs['depthmap'].sum()), 3), 'reset_color': int(obs['colormap'].sum())}]
    for i in range(steps):
        action = scripted_action(env.obs['classmap'], i)
        obs, reward, terminated, truncated, info = env.step(action)
        positions = sorted([[round(float(v), 4) for v in pos] for pos, _ in env.sim.scene.poses().values()])
        log.append({'step': i, 'action': [action['primitive'], list(action['pix'])], 'reward': reward,
                    'executed': bool(info['executed']), 'attempted': bool(info['attempted']),
                    'sorted': bool(info['sorted']), 'depth': round(float(obs['depthmap'].sum()), 3),
                    'color': int(obs['colormap'].sum()), 'positions': positions})
        if terminated or truncated:
            break
    return log


def compare(a, b, tol=2e-3):
    """Returns a list of differences between two scenario logs."""
    diffs = []
    if len(a) != len(b):
        return ['different number of steps: %d vs %d' % (len(a), len(b))]
    for i, (x, y) in enumerate(zip(a, b)):
        for key in x:
            if key == 'positions':
                if len(x[key]) != len(y[key]) or not np.allclose(x[key], y[key], atol=tol):
                    diffs.append('step %d: object positions differ' % i)
            elif isinstance(x[key], float) or key in ('depth', 'reset_depth'):
                if abs(x[key] - y[key]) > max(1e-2, 1e-3 * abs(x[key])):
                    diffs.append('step %d: %s %s vs %s' % (i, key, x[key], y[key]))
            elif x[key] != y[key]:
                diffs.append('step %d: %s %s vs %s' % (i, key, x[key], y[key]))
    return diffs


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'check'
    qualities = sys.argv[2:] or ['legacy', 'modern']
    for q in qualities:
        path = os.path.join(GOLDEN, q + '.json')
        log = run(q)
        if mode == 'record':
            json.dump(log, open(path, 'w'), indent=1)
            print('recorded', path, [(s.get('reward'), s.get('executed')) for s in log[1:]])
        else:
            diffs = compare(json.load(open(path)), log)
            print(q, 'OK' if not diffs else 'DIFFERENT', *diffs, sep='\n  ')
