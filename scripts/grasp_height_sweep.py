"""Does the suction contact height decide whether a pick tips objects over? Oracle picks of the front-most object in full scenes at
the original fixed height (4 cm above the floor) and at fractions of the object's own height.

    python scripts/grasp_height_sweep.py --scenes 30 --objects 5 [--out runs/grasp_height.json]

Per setting: held after the lift, scene invalid afterwards, and whether the toppled object was the picked one or a neighbour; per
object type."""
import argparse
import json
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from l2r.config import SimParams  # noqa: E402
from l2r.sim.environment import Environment  # noqa: E402
from l2r.sim.primitives import PICK_HEIGHT  # noqa: E402

FRACTIONS = (0.2, 0.35, 0.5, 0.65)


def reset(env, seed):
    np.random.seed(seed)
    random.seed(seed)
    assert env.reset()
    p = env.p
    ids = [i for i in env.scene.object_ids if abs(p.getBasePositionAndOrientation(i)[0][1]) < 0.2] or list(env.scene.object_ids)
    uid = max(ids, key=lambda i: p.getBasePositionAndOrientation(i)[0][0])  # front-most: reachable without clearing others
    lo, hi = p.getAABB(uid)
    return uid, np.asarray(p.getBasePositionAndOrientation(uid)[0]), hi[2] - env.ws.bounds[2][0], env.scene.names[uid].split()[-1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--scenes', type=int, default=30)
    parser.add_argument('--objects', type=int, default=5)
    parser.add_argument('--out', default='')
    args = parser.parse_args(argv)
    env = Environment(num_obj=[args.objects], params=SimParams())
    settings = [('fixed 4 cm', None)] + [('%d%% of height' % (100 * f), f) for f in FRACTIONS]
    rows = []
    for seed in range(args.scenes):
        for name, fraction in settings:
            uid, centre, top, kind = reset(env, seed)
            height = None if fraction is None else float(np.clip(fraction * top, 0.02, top - 0.02))
            held = bool(env.pick(centre[:2], height=height)) if height is not None else bool(env.pick(centre[:2]))
            toppled = env.scene.invalid_objects()
            rows.append({'seed': seed, 'setting': name, 'type': kind, 'object_height': round(float(top), 3),
                         'contact_height': PICK_HEIGHT if height is None else round(height, 3), 'held': held,
                         'invalid': bool(toppled), 'toppled_picked': uid in toppled,
                         'toppled_other': any(t != uid for t in toppled)})
        print('scene %d done' % seed, flush=True)

    def table(group):
        out = {}
        for name, _ in settings:
            sel = [r for r in group if r['setting'] == name]
            if sel:
                out[name] = {k: round(float(np.mean([r[k] for r in sel])), 3) for k in ('held', 'invalid', 'toppled_picked', 'toppled_other')}
                out[name]['n'] = len(sel)
        return out

    by_type = defaultdict(list)
    for r in rows:
        by_type[r['type']].append(r)
    summary = {'all': table(rows), **{kind: table(group) for kind, group in sorted(by_type.items())}}
    for group, result in summary.items():
        print('\n%s\n| setting | held | invalid | toppled picked | toppled other | n |\n|---|---|---|---|---|---|' % group)
        for name, m in result.items():
            print('| %s | %.2f | %.2f | %.2f | %.2f | %d |' % (name, m['held'], m['invalid'], m['toppled_picked'], m['toppled_other'], m['n']))
    if args.out:
        with open(args.out, 'w') as f:
            json.dump({'summary': summary, 'rows': rows}, f, indent=1)


if __name__ == '__main__':
    main()
