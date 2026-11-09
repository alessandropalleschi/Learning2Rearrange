"""Evaluate a trained policy (greedy) on fixed seeds, per object count.

    python evaluate.py --snapshot logs/train/models/snapshot-0010000.pth --objects 3 5 8 --episodes 20 --num_envs 4

Prints and saves (logs/<dir>/eval.json): success rate, mean actions (successful episodes) and mean return.
"""
import argparse
import json
import os
from functools import partial

import numpy as np

import env as sim_module
from config import Config, SimParams
from policy import attach_initial, carry, select_action
from rearrange_env import make_env
from trainer import Trainer
from utils import seed_everything
from vec_env import VecEnv


def evaluate(trainer, venv, cfg, num_obj, episodes, base_seed):
    """Runs `episodes` greedy episodes (seeds base_seed..base_seed+episodes-1) spread over the parallel envs."""
    n = venv.num_envs
    seeds = list(range(base_seed, base_seed + episodes))
    results, active = [], [None] * n  # active[i] = seed running in env i
    obs = [None] * n
    reset = lambda i: attach_initial(trainer, venv.reset(num_obj, seeds=[active[i]], indices=[i])[0])
    for i in range(min(n, len(seeds))):
        active[i] = seeds.pop(0)
        obs[i] = reset(i)
    while any(a is not None for a in active):
        idx = [i for i in range(n) if active[i] is not None]
        q = trainer.forward_batch([obs[i] for i in idx])
        h_new = q.hidden
        actions = [None] * n
        for k, i in enumerate(idx):
            p, pix, _ = select_action(trainer, obs[i], q.push[k], q.pick[k], q.place[k], 0.0, cfg)
            actions[i] = {'primitive': p, 'pix': pix}
        # inactive envs get a dummy action; their result is ignored
        for i in range(n):
            if actions[i] is None:
                actions[i] = {'primitive': 'push', 'pix': (0, 0, 0)}
        next_obs, rewards, term, trunc, infos = venv.step(actions)
        for k, i in enumerate(idx):
            obs[i] = carry(trainer, next_obs[i], None if h_new is None else h_new[k], actions[i]['primitive'],
                           actions[i]['pix'], infos[i]['place_pix'])
            if term[i] or trunc[i]:
                results.append({'seed': active[i], 'success': bool(infos[i]['sorted']),
                                'actions': infos[i]['actions'], 'return': infos[i]['episode_return']})
                if seeds:
                    active[i] = seeds.pop(0)
                    obs[i] = reset(i)
                else:
                    active[i] = None
    ok = [r for r in results if r['success']]
    return {'num_obj': num_obj, 'episodes': len(results), 'success_rate': len(ok) / max(len(results), 1),
            'mean_actions_success': float(np.mean([r['actions'] for r in ok])) if ok else None,
            'mean_return': float(np.mean([r['return'] for r in results]))}


def main(args):
    cfg = Config(max_actions=args.max_actions, belief=args.belief, placement=args.placement)
    params = SimParams.preset(args.physics)
    params.step_distance, params.move_joints_max_steps = args.step_distance, args.move_budget
    sim_module.configure_cabinet(params)
    seed_everything(args.seed)
    trainer = Trainer(cfg, is_testing=True, load_snapshot=True, snapshot=args.snapshot)
    venv = VecEnv([partial(make_env, cfg, False, 1, params, args.seed + r) for r in range(args.num_envs)])
    summary = []
    try:
        for n in args.objects:
            res = evaluate(trainer, venv, cfg, n, args.episodes, args.seed)
            print(json.dumps(res))
            summary.append(res)
    finally:
        venv.close()
    out = os.path.join(os.path.abspath('logs'), args.logging_directory)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'eval.json'), 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a trained rearrangement policy.')
    parser.add_argument('--snapshot', required=True)
    parser.add_argument('--objects', type=int, nargs='+', default=[3, 5, 8])
    parser.add_argument('--episodes', type=int, default=20, help='episodes per object count')
    parser.add_argument('--num_envs', type=int, default=1)
    parser.add_argument('--max_actions', type=int, default=40)
    parser.add_argument('--placement', choices=['heuristic', 'learned'], default='heuristic',
                        help='must match the snapshot')
    parser.add_argument('--belief', choices=['none', 'spatial', 'recurrent'], default='none',
                        help='must match the belief the snapshot was trained with')
    parser.add_argument('--step_distance', type=float, default=0.002,
                        help='m between IK waypoints of straight moves (larger = faster, coarser; try 0.004)')
    parser.add_argument('--move_budget', type=int, default=1000, help='simulation steps allowed per joint move')
    parser.add_argument('--physics', choices=['legacy', 'paper'], default='legacy')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--logging_directory', default='eval')
    main(parser.parse_args())
