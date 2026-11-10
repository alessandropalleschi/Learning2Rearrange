"""What the parameterised primitives can do, with privileged (oracle) parameters: one object per scene.

* grasp: straight-in approach vs the approach aligned with the normal of the object's side facing the robot
  (the cup seals only on surfaces facing it; a rotated box shows no face to the straight-in cup);
* pull: attach and drag towards the robot;
* push: the tip pushes the object along several headings (towards +y, the back, -y and in between).

    python scripts/primitive_profile.py --scenes 20 [--end_effector gripper] [--out runs/primitive_profile.json]
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from l2r.config import SimParams  # noqa: E402
from l2r.sim.environment import Environment  # noqa: E402
from l2r.sim.primitives import MAX_APPROACH  # noqa: E402

HEADINGS = {'+y': np.pi / 2, 'back-left': 3 * np.pi / 4, 'back': np.pi, 'back-right': -3 * np.pi / 4, '-y': -np.pi / 2}
PUSH_HEIGHT_Z = 0.45  # the tip's height above the ground (floor 0.40 + 0.05)


def reset(env, seed):
    np.random.seed(seed)
    random.seed(seed)
    assert env.reset()
    uid = list(env.scene.object_ids)[0]
    return uid, np.asarray(env.p.getBasePositionAndOrientation(uid)[0])


def facing_side_approach(env, uid):
    """The approach angle along the normal of the object's side that faces the robot most (oracle, from the true yaw)."""
    yaw = env.p.getEulerFromQuaternion(env.p.getBasePositionAndOrientation(uid)[1])[2]
    normals = [yaw + k * np.pi / 2 for k in range(4)]
    best = max(normals, key=np.cos)  # outward normal closest to +x (towards the robot)
    return float(np.arctan2(np.sin(best), np.cos(best)))


def face_point(env, centre, direction):
    """Where a ray along `direction` towards the object centre (tip height) first hits something."""
    start = np.array([centre[0], centre[1], PUSH_HEIGHT_Z]) - 0.4 * np.array([direction[0], direction[1], 0.0])
    hit = env.p.rayTest(tuple(start), (centre[0], centre[1], PUSH_HEIGHT_Z))[0]
    return None if hit[0] < 0 else np.asarray(hit[3][:2])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--scenes', type=int, default=20)
    parser.add_argument('--end_effector', default='suction', choices=['suction', 'gripper'])
    parser.add_argument('--out', default='')
    args = parser.parse_args(argv)
    env = Environment(num_obj=[1], params=SimParams(end_effector=args.end_effector))
    p = env.p
    grasp = {'straight': [], 'aligned': [], 'angle_deg': []}
    pull = {'success': [], 'dx': [], 'valid': []}
    push = {name: {'executed': [], 'along': [], 'valid': []} for name in HEADINGS}
    for seed in range(args.scenes):
        uid, centre = reset(env, seed)
        approach = facing_side_approach(env, uid)
        grasp['straight'].append(bool(env.pick(centre[:2])))
        uid, centre = reset(env, seed)
        aligned = approach if abs(approach) <= MAX_APPROACH else 0.0
        grasp['aligned'].append(bool(env.pick(centre[:2], approach=aligned)))
        grasp['angle_deg'].append(float(np.degrees(aligned)))

        uid, centre = reset(env, seed)
        ok = env.pull(centre[:2], heading=0.0, distance=0.08)
        after = np.asarray(p.getBasePositionAndOrientation(uid)[0])
        pull['success'].append(bool(ok))
        pull['dx'].append(float(after[0] - centre[0]))
        pull['valid'].append(bool(env.scene.is_valid()))

        for name, heading in HEADINGS.items():
            uid, centre = reset(env, seed)
            direction = np.array([np.cos(heading), np.sin(heading)])
            face = face_point(env, centre, direction)
            if face is None:
                continue
            start = face - 0.02 * direction
            ok = env.push(tuple(start), push_length=0.08, heading=heading)
            after = np.asarray(p.getBasePositionAndOrientation(uid)[0])
            push[name]['executed'].append(bool(ok))
            push[name]['along'].append(float((after[:2] - centre[:2]) @ direction))
            push[name]['valid'].append(bool(env.scene.is_valid()))
        print('scene %d done' % seed, flush=True)

    rate = lambda xs: float(np.mean(xs)) if xs else float('nan')
    summary = {
        'end_effector': args.end_effector, 'scenes': args.scenes,
        'grasp straight': rate(grasp['straight']), 'grasp aligned': rate(grasp['aligned']),
        'mean |approach| deg': float(np.mean(np.abs(grasp['angle_deg']))),
        'pull success': rate(pull['success']), 'pull dx (m)': float(np.median(pull['dx'])), 'pull valid': rate(pull['valid']),
    }
    for name, r in push.items():
        summary['push %s: moved >= 2 cm' % name] = rate([a >= 0.02 for a in r['along']])
        summary['push %s: median along (m)' % name] = float(np.median(r['along'])) if r['along'] else float('nan')
        summary['push %s: valid' % name] = rate(r['valid'])
    print('\n| measure | value |\n|---|---|')
    for key, value in summary.items():
        print('| %s | %s |' % (key, '%.3f' % value if isinstance(value, float) else value))
    if args.out:
        with open(args.out, 'w') as f:
            json.dump({'summary': summary, 'grasp': grasp, 'pull': pull, 'push': push}, f, indent=1)


if __name__ == '__main__':
    main()
