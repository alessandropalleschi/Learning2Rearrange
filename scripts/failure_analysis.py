"""Why do greedy episodes fail? Runs a trained snapshot (or the scripted expert) on fresh seeds, logs every action and what it did
to the scene, and classifies each failed episode.

    python scripts/failure_analysis.py --snapshot logs/rl_object_belief/models/snapshot-0011000.pth --episodes 50 --objects 5 \\
        --out runs/failures_object.json -- <the training flags of the run, e.g. --agent object --hierarchy --belief spatial ...>
    python scripts/failure_analysis.py --expert --episodes 50 --objects 5 -- <same flags>

Outcome of an episode: sorted | invalid (an object fell / tipped; the primitive that caused it is recorded) | out of actions.
An out-of-actions episode is further labelled by what dominated it: 'no effect' (most actions moved nothing), 'failed grasps',
'hidden objects' (misplaced objects invisible at the end), 'looping' (the same action repeated), or 'slow' (actions had effects but
did not finish)."""
import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from l2r.config import Config, SimParams  # noqa: E402
from l2r.envs.rearrange import make_env  # noqa: E402
from l2r.training import add_arguments  # noqa: E402

MOVED = 0.01  # m: an object displaced by more than this was affected by the action
VISIBLE_CELLS = 6


def scene_state(env):
    """{object id: (position, in its own region, visible)} from the simulator (privileged: analysis only)."""
    scene, obs = env.sim.scene, env.obs
    out = {}
    for object_id, (pos, _) in scene.poses().items():
        low, high = scene.regions[scene.class_name(object_id)]
        out[object_id] = (np.asarray(pos), bool(low <= pos[1] <= high), int((obs['idmap'] == object_id).sum()) >= VISIBLE_CELLS)
    return out


def run_episode(env, seed, obs, act):
    """One episode from the (already reset) observation `obs`; `act(obs) -> (primitive, pix)`."""
    steps, info = [], {'sorted': False}
    terminated = truncated = False
    held = None
    while not (terminated or truncated):
        before = scene_state(env)
        chosen = act(obs)
        primitive, pix = chosen[0], chosen[1]
        q_value = chosen[2] if len(chosen) > 2 else None
        target = int(env.obs['idmap'][pix[1], pix[2]]) if primitive in ('grasp', 'push') else None
        if primitive == 'grasp':
            held = target
        obs, reward, terminated, truncated, info = env.step({'primitive': primitive, 'pix': pix})
        after = scene_state(env)
        moved = [i for i in before if i in after and np.linalg.norm(after[i][0][:2] - before[i][0][:2]) > MOVED]
        steps.append({'primitive': primitive, 'pix': [float(v) for v in pix], 'executed': bool(info['executed']),
                      'attempted': bool(info.get('attempted', True)), 'phase': info.get('phase', 'act'),
                      'target_misplaced': None if not target or target not in before else not before[target][1],
                      'moved': len(moved), 'misplaced_before': sum(not v[1] for v in before.values()),
                      'misplaced_after': sum(not v[1] for v in after.values()),
                      'hidden_misplaced_after': sum((not v[1]) and (not v[2]) for v in after.values()),
                      'invalid': bool(info.get('invalid', False)),
                      'invalid_objects': env.sim.scene.invalid_objects() if info.get('invalid') else [],
                      'object_heights': {int(i): round(float(np.subtract(*env.sim.p.getAABB(i)[::-1])[2]), 3)
                                         for i in (env.sim.scene.invalid_objects() if info.get('invalid') else [])},
                      'target': target, 'held': held,
                      'placed_in_region': (bool(after[held][1]) if held in after else None) if primitive == 'place' else None,
                      'q': q_value, 'reward': float(reward), 'terminal': bool(terminated)})
    gamma = env.cfg.gamma
    future = 0.0  # realised return from each step (as the trainer discounts: the grasp -> place step is not discounted)
    for s in reversed(steps):
        discount = 1.0 if s['phase'] == 'place' else gamma
        future = s['reward'] + (0.0 if s['terminal'] else discount * future)
        s['return'] = future
    acts = [s for s in steps if s['primitive'] != 'place']
    if info['sorted']:
        outcome, reason = 'sorted', ''
    elif info.get('invalid'):
        outcome, reason = 'invalid', steps[-1]['primitive'] if steps else ''
    else:
        outcome = 'out of actions'
        no_effect = np.mean([s['moved'] == 0 for s in acts]) if acts else 0
        failed_grasps = np.mean([s['primitive'] == 'grasp' and not s['executed'] for s in acts]) if acts else 0
        repeats = max(Counter((s['primitive'], tuple(s['pix'][:3])) for s in acts).values()) if acts else 0
        if steps and steps[-1]['hidden_misplaced_after'] > 0:
            reason = 'hidden objects'
        elif repeats >= 3:
            reason = 'looping'
        elif failed_grasps >= 0.3:
            reason = 'failed grasps'
        elif no_effect >= 0.5:
            reason = 'no effect'
        else:
            reason = 'slow'
    return {'seed': seed, 'outcome': outcome, 'reason': reason, 'actions': len(acts), 'steps': steps}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    own, flags = (argv[:argv.index('--')], argv[argv.index('--') + 1:]) if '--' in argv else (argv, [])
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--snapshot', default='')
    parser.add_argument('--expert', action='store_true')
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--objects', type=int, default=5)
    parser.add_argument('--base_seed', type=int, default=20000, help='fresh seeds, disjoint from training and validation')
    parser.add_argument('--out', default='')
    args = parser.parse_args(own)
    run_parser = argparse.ArgumentParser()
    add_arguments(run_parser)
    run = run_parser.parse_args(flags)
    cfg, params = Config.from_args(run), SimParams.from_args(run)
    cfg.max_actions = run.max_actions
    env = make_env(cfg, False, args.objects, params, args.base_seed)

    if args.expert:
        from l2r.demos import ScriptedExpert

        def run_one(seed):
            obs = env.reset(num_obj=args.objects, seed=seed)
            expert = ScriptedExpert(env)

            def act(_):
                choice = expert.act()
                if choice is None:  # nothing sensible: a push at the corner that touches nothing (costs an action)
                    return 'push', ((0, 0, 0, 0.5) if cfg.hierarchy else (0, 0, 0))
                return choice
            return run_episode(env, seed, obs, act)
    else:
        from l2r.learning.factory import make_trainer
        from l2r.learning.policy import attach_initial, select_for
        from l2r.sim.workspace import Workspace
        trainer = make_trainer(cfg, is_testing=True, load_snapshot=True, snapshot=args.snapshot)
        ws = Workspace.from_params(params)

        def act(obs):
            q = trainer.forward_batch([obs])
            primitive, pix, value = select_for(trainer, obs, q, 0, 0.0, cfg, ws)
            return primitive, pix, value

        def run_one(seed):
            obs = attach_initial(trainer, env.reset(num_obj=args.objects, seed=seed))
            return run_episode(env, seed, obs, act)

    episodes = []
    for k in range(args.episodes):
        episodes.append(run_one(args.base_seed + k))
        e = episodes[-1]
        print('episode %d: %s %s (%d actions)' % (k, e['outcome'], e['reason'], e['actions']), flush=True)

    outcomes = Counter((e['outcome'], e['reason']) for e in episodes)
    acts = [s for e in episodes for s in e['steps'] if s['primitive'] != 'place']
    places = [s for e in episodes for s in e['steps'] if s['primitive'] == 'place']
    every = [s for e in episodes for s in e['steps']]
    summary = {
        'episodes': len(episodes), 'success': float(np.mean([e['outcome'] == 'sorted' for e in episodes])),
        'outcomes': {'%s%s' % (o, ' / ' + r if r else ''): n for (o, r), n in outcomes.most_common()},
        'actions per episode': float(np.mean([e['actions'] for e in episodes])),
        'primitive mix': dict(Counter(s['primitive'] for s in acts)),
        'grasp executed': float(np.mean([s['executed'] for s in acts if s['primitive'] == 'grasp'] or [np.nan])),
        'grasp on a misplaced object': float(np.mean([bool(s['target_misplaced']) for s in acts if s['primitive'] == 'grasp'] or [np.nan])),
        'push moved something': float(np.mean([s['moved'] > 0 for s in acts if s['primitive'] == 'push'] or [np.nan])),
        'actions that moved nothing': float(np.mean([s['moved'] == 0 for s in acts] or [np.nan])),
        'steps that reduced the misplaced count': float(np.mean([s['misplaced_after'] < s['misplaced_before'] for s in every] or [np.nan])),
        'steps that increased it': float(np.mean([s['misplaced_after'] > s['misplaced_before'] for s in every] or [np.nan])),
        'invalid: the grasped/held object itself': sum(1 for s in every if s['invalid'] and s['held'] in s['invalid_objects']),
        'invalid: another object': sum(1 for s in every if s['invalid'] and s['invalid_objects'] and s['held'] not in s['invalid_objects']),
        'invalid: by primitive': dict(Counter(s['primitive'] for s in every if s['invalid'])),
        'placements': len(places),
        'placed in its own region': float(np.mean([s['placed_in_region'] for s in places if s['placed_in_region'] is not None] or [np.nan])),
    }
    valued = [s for s in every if s.get('q') is not None]
    if valued:  # overestimation: predicted Q of the chosen action minus the return that followed
        bias = lambda sel: float(np.mean([s['q'] - s['return'] for s in sel])) if sel else float('nan')
        summary['Q - return (all)'] = bias(valued)
        summary['Q - return (finished episodes only)'] = bias([s for e in episodes if e['outcome'] != 'out of actions'
                                                               for s in e['steps'] if s.get('q') is not None])
        for name in ('grasp', 'push', 'place'):
            summary['Q - return (%s)' % name] = bias([s for s in valued if s['primitive'] == name])
        summary['grasp on a placed object: Q - return'] = bias([s for s in valued if s['primitive'] == 'grasp'
                                                                and s['target_misplaced'] is False])
        summary['mean Q of the chosen actions'] = float(np.mean([s['q'] for s in valued]))
        summary['mean realised return'] = float(np.mean([s['return'] for s in valued]))
    print(json.dumps(summary, indent=1))
    if args.out:
        with open(args.out, 'w') as f:
            json.dump({'summary': summary, 'episodes': episodes}, f, indent=1)


if __name__ == '__main__':
    main()
