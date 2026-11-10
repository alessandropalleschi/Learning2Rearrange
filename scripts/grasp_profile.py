"""How does grasp success depend on WHERE on the front of an object the cup touches it?

For one object type, scenes with a single object of that type are rebuilt from fixed seeds and the suction grasp is tried at positions
across the object's width (0 = one edge, 0.5 = centre, 1 = the other edge), always at the object's own front-most cell of that row
(aim: `GRASP_DEPTH_CELLS` inside it). Prints the success rate per position: this is what the equivalence-class targets assume to
be flat.

    PYTHONPATH=. python scripts/grasp_profile.py CrackerBox --scenes 6
"""
import argparse
from types import SimpleNamespace

import numpy as np

from l2r.config import Config, SimParams
from l2r.demos import GRASP_DEPTH_CELLS
from l2r.envs.rearrange import RearrangeEnv

POSITIONS = (0.0, 0.15, 0.3, 0.5, 0.7, 0.85, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('object_type')
    parser.add_argument('--scenes', type=int, default=6)
    args = parser.parse_args()
    flags = SimpleNamespace(physics='legacy', randomize=False, step_distance=0.002, move_budget=1000, sim_quality='legacy',
                            camera_scale=1.0, force_limit=300.0, force_sense_x=None, adaptive_step=True, velocity_scale=1.0,
                            joint_tolerance=3e-3)
    cfg = Config(max_actions=60, placement='heuristic', reward_mode='sparse')
    env = RearrangeEnv(cfg, num_obj=1, params=SimParams.from_args(flags))
    results, seed, found = {u: [] for u in POSITIONS}, 5000, 0
    while found < args.scenes:
        seed += 1
        env.reset(num_obj=1, seed=seed)
        scene = env.sim.scene
        if scene.names[scene.object_ids[0]].split(' ')[1] != args.object_type:
            continue
        found += 1
        rows, cols = np.nonzero(env.obs['idmap'] == scene.object_ids[0])
        lo, hi = rows.min(), rows.max()
        width_cm = (hi - lo + 1) * 0.5
        for u in POSITIONS:
            row = int(round(lo + u * (hi - lo)))
            front = cols[rows == row].max() if (rows == row).any() else None
            if front is None:
                continue
            env.reset(num_obj=1, seed=seed)  # a clean scene for every attempt
            col = max(front - GRASP_DEPTH_CELLS, int(cols[rows == row].min()))
            ok = env.sim.pick(env.ws.px_to_xy(col, row))
            results[u].append(bool(ok))
        print('scene %d (width %.1f cm): %s' % (found, width_cm, ''.join('#' if r[-1] else '.' for r in results.values() if r)), flush=True)
    print('%s: success by position across the width (edge ... centre ... edge)' % args.object_type)
    print('  position ', ' '.join('%5.2f' % u for u in POSITIONS))
    print('  success  ', ' '.join('%5.2f' % np.mean(results[u]) if results[u] else '  n/a' for u in POSITIONS), ' (n = %d per position)' % args.scenes)


if __name__ == '__main__':
    main()
