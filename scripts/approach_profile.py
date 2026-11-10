"""Straight-in grasps vs the clearest approach line (`--grasp_approach clear`) in full scenes: every visible object is grasped once
each way (after a reset of the same scene), at the front-face cell the object agent would use; the target comes from the
observation (no privileged information).

    python scripts/approach_profile.py --scenes 30 --objects 5 [--out runs/approach_profile.json]"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from l2r.config import SimParams  # noqa: E402
from l2r.envs.approach import choose_approach, obstacles  # noqa: E402
from l2r.learning.object_centric import GRASP_DEPTH_CELLS, extract_instances  # noqa: E402
from l2r.sim.environment import Environment  # noqa: E402


def reset(env, seed):
    np.random.seed(seed)
    random.seed(seed)
    assert env.reset()
    return env.observe()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--scenes', type=int, default=30)
    parser.add_argument('--objects', type=int, default=5)
    parser.add_argument('--out', default='')
    args = parser.parse_args(argv)
    env = Environment(num_obj=[args.objects], params=SimParams())
    rows = []
    for seed in range(args.scenes):
        obs = reset(env, seed)
        targets = []
        for inst in extract_instances(obs):
            col = int(inst.cols.max()) - GRASP_DEPTH_CELLS
            at = inst.rows[inst.cols == col] if (inst.cols == col).any() else inst.rows
            targets.append((int(np.median(at)), col))
        for target in targets:
            for mode in ('straight', 'clear'):
                obs = reset(env, seed)
                approach, count = (0.0, obstacles(obs, target, 0.0)) if mode == 'straight' else choose_approach(obs, target)
                xy = env.ws.px_to_xy(target[1], target[0])
                held = bool(env.pick(xy, approach=approach))
                toppled = env.scene.invalid_objects()
                rows.append({'seed': seed, 'target': target, 'mode': mode, 'approach_deg': round(float(np.degrees(approach)), 1),
                             'obstacles': count, 'blocked_straight': obstacles(reset(env, seed), target, 0.0) > 0,
                             'held': held, 'invalid': bool(toppled)})
        print('scene %d: %d targets' % (seed, len(targets)), flush=True)

    def summary(sel):
        out = {}
        for mode in ('straight', 'clear'):
            r = [x for x in sel if x['mode'] == mode]
            if r:
                out[mode] = {'held': round(float(np.mean([x['held'] for x in r])), 3), 'invalid': round(float(np.mean([x['invalid'] for x in r])), 3),
                             'mean obstacles': round(float(np.mean([x['obstacles'] for x in r])), 1),
                             'mean |approach| deg': round(float(np.mean([abs(x['approach_deg']) for x in r])), 1), 'n': len(r)}
        return out
    result = {'all': summary(rows), 'straight corridor blocked': summary([r for r in rows if r['blocked_straight']]),
              'straight corridor free': summary([r for r in rows if not r['blocked_straight']])}
    for group, table in result.items():
        print('\n%s\n| mode | held | invalid | mean obstacles | mean abs approach | n |\n|---|---|---|---|---|---|' % group)
        for mode, m in table.items():
            print('| %s | %.2f | %.2f | %.1f | %.1f | %d |' % (mode, m['held'], m['invalid'], m['mean obstacles'], m['mean |approach| deg'], m['n']))
    if args.out:
        with open(args.out, 'w') as f:
            json.dump({'summary': result, 'rows': rows}, f, indent=1)


if __name__ == '__main__':
    main()
