"""Evaluate a trained policy (greedy) on fixed seeds, per object count.

    python evaluate.py --snapshot logs/train/models/snapshot-0010000.pth --objects 3 5 8 --episodes 20 --num_envs 4

Prints and saves (logs/<dir>/eval.json): success rate, mean actions (successful episodes) and mean return.
"""
import argparse
import json
import os
from functools import partial

import numpy as np

from l2r.config import Config, SimParams
from l2r.envs.rearrange import make_env
from l2r.envs.vec import VecEnv
from l2r.learning.policy import attach_initial, carry, select_for
from l2r.learning.factory import make_trainer
from l2r.sim.workspace import Workspace
from l2r.utils import seed_everything

IDLE_ACTION = {'primitive': 'push', 'pix': (0, 0, 0, 0.0)}  # sent to simulators that have no episode left (ignored)


def evaluate(trainer, venv, cfg, ws, num_obj, episodes, base_seed):
    """Runs `episodes` greedy episodes (seeds base_seed..base_seed+episodes-1) spread over the parallel envs."""
    n = venv.num_envs
    pending = list(range(base_seed, base_seed + episodes))
    seed_of, obs, results = [None] * n, [None] * n, []

    def start(i):
        seed_of[i] = pending.pop(0)
        obs[i] = attach_initial(trainer, venv.reset(num_obj, seeds=[seed_of[i]], indices=[i])[0])

    for i in range(min(n, len(pending))):
        start(i)
    while any(seed is not None for seed in seed_of):
        active = [i for i in range(n) if seed_of[i] is not None]
        q = trainer.forward_batch([obs[i] for i in active])
        actions = [IDLE_ACTION] * n
        for k, i in enumerate(active):
            primitive, pix, _ = select_for(trainer, obs[i], q, k, 0.0, cfg, ws)
            actions[i] = {'primitive': primitive, 'pix': pix}
        next_obs, _, terminated, truncated, infos = venv.step(actions)
        for k, i in enumerate(active):
            obs[i] = carry(trainer, next_obs[i], None if q.hidden is None else q.hidden[k], actions[i]['primitive'],
                           actions[i]['pix'], infos[i]['place_pix'])
            if terminated[i] or truncated[i]:
                results.append({'seed': seed_of[i], 'success': bool(infos[i]['sorted']), 'actions': infos[i]['actions'],
                                'return': infos[i]['episode_return']})
                seed_of[i] = None
                if pending:
                    start(i)
    ok = [r for r in results if r['success']]
    return {'num_obj': num_obj, 'episodes': len(results), 'success_rate': len(ok) / max(len(results), 1),
            'mean_actions_success': float(np.mean([r['actions'] for r in ok])) if ok else None,
            'mean_return': float(np.mean([r['return'] for r in results]))}


def main(argv=None):
    parser = argparse.ArgumentParser(description='Evaluate a trained rearrangement policy.')
    parser.add_argument('--snapshot', required=True)
    parser.add_argument('--objects', type=int, nargs='+', default=[3, 5, 8])
    parser.add_argument('--episodes', type=int, default=20, help='episodes per object count')
    parser.add_argument('--num_envs', type=int, default=1)
    parser.add_argument('--max_actions', type=int, default=40)
    Config.add_arguments(parser)  # must match the snapshot
    SimParams.add_arguments(parser)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--logging_directory', default='eval')
    args = parser.parse_args(argv)

    cfg = Config.from_args(args)
    params = SimParams.from_args(args)
    ws = Workspace.from_params(params)
    seed_everything(args.seed)
    trainer = make_trainer(cfg, is_testing=True, load_snapshot=True, snapshot=args.snapshot)
    venv = VecEnv([partial(make_env, cfg, False, 1, params, args.seed + r) for r in range(args.num_envs)])
    summary = []
    try:
        for num_obj in args.objects:
            result = evaluate(trainer, venv, cfg, ws, num_obj, args.episodes, args.seed)
            print(json.dumps(result))
            summary.append(result)
    finally:
        venv.close()
    out = os.path.join(os.path.abspath('logs'), args.logging_directory)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'eval.json'), 'w') as f:
        json.dump(summary, f, indent=2)
