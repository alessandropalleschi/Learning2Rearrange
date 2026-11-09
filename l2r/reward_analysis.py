"""What does the reward signal look like?  Random actions on the real simulator, measured per action:

  * how often an action changes the number of correctly placed objects (the only signal of the sparse reward),
  * the potential-based shaping terms ('progress', 'distance'),
  * actions that moved nothing (what the 'noop' variant penalises),
  * how many objects an action moves and how much of the scene changes -- the volatility of the scene decides how long a
    plan made from one observation stays valid (relevant for action-chunking / sequence-prediction models).

    python analyze_rewards.py --actions 120 --num_obj 5
"""
import argparse

import numpy as np

from l2r.config import Config, SimParams
from l2r.envs.rearrange import RearrangeEnv
from l2r.learning.policy import select_for
from l2r.learning.trainer import Trainer
from l2r.sim.workspace import Workspace


def analyse(args):
    cfg = Config.from_args(args)
    cfg.max_actions = 10 ** 6
    params = SimParams.from_args(args)
    ws = Workspace.from_params(params)
    np.random.seed(args.seed)
    env, trainer = RearrangeEnv(cfg, num_obj=args.num_obj, params=params), Trainer(cfg, is_testing=True)
    obs, rows = env.reset(), []
    for _ in range(args.actions):
        before = env.sim.scene.progress(), env.sim.scene.positions()
        heights = env.obs['depthmap'].copy()
        primitive, pix, _ = select_for(trainer, obs, trainer.forward_batch([obs]), 0, 1.0, cfg, ws)
        obs, reward, terminated, truncated, info = env.step({'primitive': primitive, 'pix': pix})
        if info.get('phase') == 'place':  # learned placement: wait for the complete pick-and-place
            continue
        after = env.sim.scene.progress(), env.sim.scene.positions()
        shift = np.linalg.norm(after[1] - before[1], axis=1) if len(after[1]) == len(before[1]) else np.zeros(1)
        rows.append({'primitive': primitive, 'executed': info['executed'], 'reward': reward, 'shaping': info['shaping'],
                     'd_sorted': after[0][0] - before[0][0], 'd_distance': before[0][1] - after[0][1],
                     'moved': int((shift > 0.01).sum()), 'noop': bool((shift < 0.005).all()),
                     'changed_cells': float((np.abs(env.obs['depthmap'] - heights) > 0.01).mean()), 'invalid': info['invalid']})
        if terminated or truncated:
            obs = env.reset()
    return rows


def report(rows):
    n, col = len(rows), lambda k: np.array([r[k] for r in rows], float)
    print('%d actions (%d push, %d pick-and-place), %.0f%% executed, %.1f%% ended in an invalid state'
          % (n, sum(r['primitive'] == 'push' for r in rows), sum(r['primitive'] != 'push' for r in rows),
             100 * col('executed').mean(), 100 * col('invalid').mean()))
    print('sparse signal: %.1f%% of the actions changed the number of correctly placed objects (+%.1f%% / -%.1f%%)'
          % (100 * (col('d_sorted') != 0).mean(), 100 * (col('d_sorted') > 0).mean(), 100 * (col('d_sorted') < 0).mean()))
    print('distance potential: %.1f%% of the actions moved objects closer to their region, %.1f%% farther; mean change %+.4f m'
          % (100 * (col('d_distance') > 1e-3).mean(), 100 * (col('d_distance') < -1e-3).mean(), col('d_distance').mean()))
    print('no-ops (no object moved): %.1f%% of the actions (%.1f%% of the executed ones)'
          % (100 * col('noop').mean(), 100 * col('noop')[col('executed') > 0].mean() if (col('executed') > 0).any() else 0))
    print('scene volatility: %.2f objects moved per action (>1 cm), %.1f%% of the actions moved two or more; '
          '%.1f%% of the heightmap cells changed by >1 cm per action'
          % (col('moved').mean(), 100 * (col('moved') >= 2).mean(), 100 * col('changed_cells').mean()))
    print('mean reward %.2f, mean shaping term %+.3f' % (col('reward').mean(), col('shaping').mean()))


def main(argv=None):
    parser = argparse.ArgumentParser(description='Measure the reward signal and the scene volatility with random actions.')
    parser.add_argument('--actions', type=int, default=120)
    parser.add_argument('--num_obj', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    Config.add_arguments(parser)
    SimParams.add_arguments(parser)
    parser.add_argument('--randomize', action='store_true', default=False)
    report(analyse(parser.parse_args(argv)))


if __name__ == '__main__':
    main()
