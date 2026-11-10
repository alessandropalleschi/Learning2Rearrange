"""Why do placements topple the placed object? Oracle pick of the front-most object, heuristic placement in its region; records the
object's tilt and its drift from the grasp offset just before the cup releases it, and whether it stands afterwards.

    python scripts/place_profile.py --scenes 30 --objects 5 [--out runs/place_profile.json]"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from l2r.config import Config, SimParams  # noqa: E402
from l2r.envs.rearrange import RearrangeEnv  # noqa: E402


def tilt(p, uid):
    roll, pitch, _ = p.getEulerFromQuaternion(p.getBasePositionAndOrientation(uid)[1])
    return float(np.degrees(max(abs(roll), abs(pitch))))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--scenes', type=int, default=30)
    parser.add_argument('--objects', type=int, default=5)
    parser.add_argument('--out', default='')
    args = parser.parse_args(argv)
    cfg = Config(placement='learned', footprint_placement=True, max_actions=10)
    params = SimParams(adaptive_step=True, joint_tolerance=3e-3)
    env = RearrangeEnv(cfg, num_obj=args.objects, params=params)
    p, cup = env.sim.p, env.sim.tool
    rows, record = [], {}
    real_release = cup.release

    def release():  # measure the held object just before the vacuum is switched off
        if cup.activated and cup.grip_body is not None:
            tool_pos, _ = env.sim.arm.tool_frame()
            pos, _ = p.getBasePositionAndOrientation(cup.grip_body)
            record.update(tilt=tilt(p, cup.grip_body), drift=float(abs(np.linalg.norm(np.subtract(pos, tool_pos)) - cup.grip_distance)),
                          height=float(pos[2] - env.ws.bounds[2][0]), x=float(pos[0]))
        return real_release()
    cup.release = release
    for seed in range(args.scenes):
        np.random.seed(seed)
        random.seed(seed)
        env.reset(num_obj=args.objects, seed=seed)
        ids = [i for i in env.sim.scene.object_ids if abs(p.getBasePositionAndOrientation(i)[0][1]) < 0.2] or list(env.sim.scene.object_ids)
        uid = max(ids, key=lambda i: p.getBasePositionAndOrientation(i)[0][0])
        idmap = env.obs['idmap']
        cells = np.argwhere(idmap == uid)
        if not len(cells):
            continue
        row, col = int(np.median(cells[:, 0])), int(cells[:, 1].max()) - 2
        obs, _, _, _, info = env.step({'primitive': 'grasp', 'pix': (0, row, col)})
        if info.get('phase') != 'place':
            continue
        lo, hi = p.getAABB(uid)
        kind = env.sim.scene.names[uid].split()[-1]
        col_t, row_t = obs['heuristic_place']
        record.clear()
        _, _, terminated, _, info = env.step({'primitive': 'place', 'pix': (0, int(row_t), int(col_t))})
        rows.append({'seed': seed, 'type': kind, 'object_height': round(float(hi[2] - lo[2]), 3), 'target_col': int(col_t),
                     'tilt_at_release': record.get('tilt'), 'drift_at_release': record.get('drift'),
                     'height_at_release': record.get('height'), 'standing_after': tilt(p, uid) < 30.0,
                     'invalid': bool(info.get('invalid', False))})
        print(rows[-1], flush=True)
    toppled = [r for r in rows if not r['standing_after']]
    standing = [r for r in rows if r['standing_after']]
    mean = lambda sel, k: float(np.mean([r[k] for r in sel if r[k] is not None])) if sel else float('nan')
    summary = {'placements': len(rows), 'toppled': len(toppled),
               'tilt at release (toppled / standing)': (mean(toppled, 'tilt_at_release'), mean(standing, 'tilt_at_release')),
               'drift at release (toppled / standing)': (mean(toppled, 'drift_at_release'), mean(standing, 'drift_at_release')),
               'height above the floor at release (toppled / standing)': (mean(toppled, 'height_at_release'), mean(standing, 'height_at_release')),
               'target column (toppled / standing)': (mean(toppled, 'target_col'), mean(standing, 'target_col')),
               'toppled by type': {k: sum(1 for r in toppled if r['type'] == k) for k in sorted({r['type'] for r in rows})},
               'placements by type': {k: sum(1 for r in rows if r['type'] == k) for k in sorted({r['type'] for r in rows})}}
    print(json.dumps(summary, indent=1))
    if args.out:
        with open(args.out, 'w') as f:
            json.dump({'summary': summary, 'rows': rows}, f, indent=1)


if __name__ == '__main__':
    main()
