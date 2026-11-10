"""Physics sanity checks of the simulator (no learning): run under each --sim_quality and compare.

1. rest: a 5-object scene left alone for 10 s: object drift and tilt (should be ~0);
2. tipping: each YCB type pushed at 5 cm (the pusher's height) and at 2/3 of its height: does it slide or tip? Compared with the
   rigid-body condition: a push at height h tips the object before it slides if h > (depth along the push / 2) / mu, with mu the
   combined object-floor friction (PyBullet multiplies the two coefficients);
3. contact: the peak normal force between the robot and the object during a push, and the object's peak speed.

    python scripts/physics_sanity.py --sim_quality legacy|modern [--out runs/physics_legacy.json]"""
import argparse
import json
import os
import random
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from l2r.config import SimParams  # noqa: E402
from l2r.sim.environment import Environment  # noqa: E402

TYPES = ('ChipsCan', 'CrackerBox', 'MustardBottle', 'PottedMeatCan')


def params_for(quality, pusher='tip'):
    flags = SimpleNamespace(physics='legacy', randomize=False, step_distance=0.002, move_budget=1000, sim_quality=quality,
                            camera_scale=1.0, force_limit=300.0, force_sense_x=None, adaptive_step=True, velocity_scale=1.0,
                            joint_tolerance=3e-3)
    params = SimParams.from_args(flags)
    params.pusher = pusher
    return params


def tilt_deg(p, uid):
    r, q, _ = p.getEulerFromQuaternion(p.getBasePositionAndOrientation(uid)[1])
    return float(np.degrees(max(abs(r), abs(q))))


def rest_test(params, seeds=5):
    env = Environment(num_obj=[5], params=params)
    p, out = env.p, []
    for seed in range(seeds):
        np.random.seed(seed)
        random.seed(seed)
        if not env.reset():
            continue
        before = {i: (np.asarray(p.getBasePositionAndOrientation(i)[0]), tilt_deg(p, i)) for i in env.scene.object_ids}
        for _ in range(int(10.0 / env.time_step)):
            p.stepSimulation()
        for i, (pos, t) in before.items():
            out.append({'drift_mm': float(1000 * np.linalg.norm(np.asarray(p.getBasePositionAndOrientation(i)[0]) - pos)),
                        'tilt_change_deg': abs(tilt_deg(p, i) - t)})
    p.disconnect()
    return {'objects': len(out), 'max drift (mm)': max(o['drift_mm'] for o in out), 'median drift (mm)': float(np.median([o['drift_mm'] for o in out])),
            'max tilt change (deg)': max(o['tilt_change_deg'] for o in out)}


def push_test(params, scenes=40):
    """Push the front-most object of single-object scenes along +y at two heights."""
    env = Environment(num_obj=[1], params=params)
    p, rows = env.p, []
    for seed in range(scenes):
        for height in ('low', 'high'):
            np.random.seed(seed)
            random.seed(seed)
            if not env.reset():
                continue
            uid = env.scene.object_ids[0]
            kind = env.scene.names[uid].split()[-1]
            lo, hi = p.getAABB(uid)
            pos = np.asarray(p.getBasePositionAndOrientation(uid)[0])
            object_height = hi[2] - lo[2]
            floor = env.ws.bounds[2][0]
            z = floor + (0.05 if height == 'low' else max(0.05, 2 * object_height / 3))
            depth_along = hi[1] - lo[1]  # extent along the push (y)
            mu = p.getDynamicsInfo(uid, -1)[1] * p.getDynamicsInfo(env.scene.wall_ids['floor'], -1)[1]
            start = (pos[0], lo[1] - 0.03)
            peak_force, peak_speed = 0.0, 0.0
            real_step = p.stepSimulation

            def step():
                nonlocal peak_force, peak_speed
                real_step()
                for c in p.getContactPoints(bodyA=env.arm.body, bodyB=uid):
                    peak_force = max(peak_force, c[9])
                peak_speed = max(peak_speed, float(np.linalg.norm(p.getBaseVelocity(uid)[0])))
            p.stepSimulation = step
            try:
                import l2r.sim.primitives as prim
                old = prim.PUSH_HEIGHT
                prim.PUSH_HEIGHT = z - floor
                env.push(start, heading=np.pi / 2, push_length=0.08)
            finally:
                prim.PUSH_HEIGHT = old
                p.stepSimulation = real_step
            moved = float(np.linalg.norm(np.asarray(p.getBasePositionAndOrientation(uid)[0])[:2] - pos[:2]))
            tipped = tilt_deg(p, uid) > 30
            rows.append({'seed': seed, 'type': kind, 'height': height, 'push_height_m': round(float(z - floor), 3),
                         'object_height_m': round(float(object_height), 3), 'depth_along_m': round(float(depth_along), 3), 'mu': round(float(mu), 3),
                         'predicted_tip': bool((z - floor) > depth_along / 2 / max(mu, 1e-6)), 'tipped': bool(tipped),
                         'moved_m': round(moved, 3), 'peak_contact_N': round(peak_force, 1), 'peak_speed_mps': round(peak_speed, 3)})
    p.disconnect()
    summary = {}
    for kind in TYPES:
        for height in ('low', 'high'):
            sel = [r for r in rows if r['type'] == kind and r['height'] == height]
            if sel:
                summary['%s %s' % (kind, height)] = {
                    'n': len(sel), 'tipped': float(np.mean([r['tipped'] for r in sel])),
                    'predicted to tip': float(np.mean([r['predicted_tip'] for r in sel])),
                    'moved (m)': float(np.median([r['moved_m'] for r in sel])),
                    'peak contact force (N)': float(np.median([r['peak_contact_N'] for r in sel])),
                    'peak object speed (m/s)': float(np.median([r['peak_speed_mps'] for r in sel]))}
    return summary, rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--sim_quality', default='legacy', choices=['legacy', 'improved', 'modern'])
    parser.add_argument('--scenes', type=int, default=40)
    parser.add_argument('--pusher', default='tip', choices=['tip', 'paddle'])
    parser.add_argument('--skip_rest', action='store_true')
    parser.add_argument('--out', default='')
    args = parser.parse_args(argv)
    params = params_for(args.sim_quality, args.pusher)
    rest = {} if args.skip_rest else rest_test(params)
    push, rows = push_test(params, args.scenes)
    print(json.dumps({'rest': rest, 'push': push}, indent=1))
    if args.out:
        with open(args.out, 'w') as f:
            json.dump({'sim_quality': args.sim_quality, 'rest': rest, 'push': push, 'rows': rows}, f, indent=1)


if __name__ == '__main__':
    main()
