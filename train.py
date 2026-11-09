"""Training / testing loop: double DQN with push + pick-and-place primitives, curriculum and HER.

Several simulators can run in parallel (--num_envs); the policy is evaluated on all of them in one batch.
"""
import argparse
import os
from collections import deque
from functools import partial

import cv2

import env as sim_module
from config import Config, SimParams
from logger import Logger
from memory import Transition
from policy import attach_initial, carry, carry_over, select_action
from rearrange_env import make_env
from trainer import Trainer
from utils import seed_everything
from vec_env import VecEnv


def main(args):
    cfg = Config(start_objects=args.num_obj, max_objects=args.max_obj, max_actions=args.max_actions,
                 belief=args.belief, placement=args.placement, eps_decay=args.eps_decay)
    params = SimParams.from_args(args)
    sim_module.configure_cabinet(params)  # the main process needs the same workspace bounds as the workers
    seed_everything(args.random_seed)
    log_dir = os.path.join(os.path.abspath('logs'), args.logging_directory)
    os.makedirs(os.path.join(log_dir, 'models'), exist_ok=True)

    n = args.num_envs
    venv = VecEnv([partial(make_env, cfg, args.gui and r == 0, cfg.start_objects, params, args.random_seed + r)
                   for r in range(n)], start_method=args.start_method)
    trainer = Trainer(cfg, is_testing=args.is_testing, load_snapshot=args.load_snapshot,
                      snapshot=args.snapshot, snapshot_target=args.snapshot_target)
    logger = Logger(log_dir)

    num_obj = cfg.start_objects
    successes, scores = deque(maxlen=cfg.success_window), deque(maxlen=cfg.success_window)
    stage_iteration, episode = 0, 0
    obs = [attach_initial(trainer, o) for o in venv.reset(num_obj, seeds=[args.random_seed + 1000 * r for r in range(n)])]
    try:
        while True:
            # epsilon-greedy: 0.8 annealed to 0.01 for each curriculum stage (paper)
            eps = 0.0 if args.is_testing else max(cfg.eps_start * cfg.eps_decay ** stage_iteration, cfg.eps_end)
            q = trainer.forward_batch(obs)
            h_new = q.hidden
            choices = [select_action(trainer, obs[i], q.push[i], q.pick[i], q.place[i], eps, cfg) for i in range(n)]
            next_obs, rewards, terminated, truncated, infos = venv.step(
                [{'primitive': c[0], 'pix': c[1]} for c in choices])

            batch = []
            for i in range(n):
                primitive, pix, _ = choices[i]
                carry(trainer, next_obs[i], None if h_new is None else h_new[i], primitive, pix, infos[i]['place_pix'])
                if args.save_visualizations and i == 0:
                    place = infos[i]['place_pix']
                    vis = trainer.get_prediction_vis({'push': q.push, 'grasp': q.pick, 'place': q.place}[primitive][i],
                                                     obs[i]['colormap'],
                                                     pix, pix if place is None else [place[1], place[0]])
                    cv2.imwrite(os.path.join(log_dir, 'visualization.%s.png' % primitive), vis)
                if args.is_testing:
                    continue
                t = Transition(obs[i], pix, next_obs[i], rewards[i], primitive, terminated[i])
                trainer.buffer.push(*t)
                batch.append(t)
                if infos[i]['relabel']:  # hindsight experience replay (paper Fig. 8)
                    obs_r, next_r = infos[i]['relabel_pair']
                    carry_over(trainer, obs[i], obs_r)
                    carry_over(trainer, next_obs[i], next_r)
                    trainer.buffer.push(obs_r, pix, next_r, cfg.reward_goal, primitive, True)

            if batch:
                trainer.train_step(batch)  # short-term memory: learn from the newest transitions at once
                for _ in range(args.updates_per_step * n):
                    loss = trainer.optimize()
                    if loss is not None and trainer.opt_steps % 10 == 0:
                        logger.scalar('train/loss', loss, trainer.iteration)
                        logger.scalar('train/q_mean', trainer.last_q, trainer.iteration)
                        logger.scalar('train/epsilon', eps, trainer.iteration)
                        logger.scalar('train/lr', trainer.optimizer.param_groups[0]['lr'], trainer.iteration)
                        logger.scalar('train/buffer', len(trainer.buffer), trainer.iteration)
                        print('Iteration %d  loss %.4f  eps %.3f' % (trainer.iteration, loss, eps))
            trainer.iteration += n
            stage_iteration += n
            if not args.is_testing and trainer.iteration // args.save_every != (trainer.iteration - n) // args.save_every:
                trainer.save(os.path.join(log_dir, 'models', 'snapshot-%07d.pth' % trainer.iteration),
                             os.path.join(log_dir, 'models', 'target-%07d.pth' % trainer.iteration))

            finished = []
            for i in range(n):
                if terminated[i] or truncated[i]:
                    finished.append(i)
                    episode += 1
                    info = infos[i]
                    successes.append(1 if info['sorted'] else 0)
                    scores.append(info['episode_return'])
                    avg_success = sum(successes) / len(successes)
                    print('Episode %d: return %.0f, success %s, avg success %.2f, avg return %.1f'
                          % (episode, info['episode_return'], info['sorted'], avg_success, sum(scores) / len(scores)))
                    logger.episode(episode, trainer.iteration, num_obj, info['episode_return'], info['actions'],
                                   info['sorted'], eps)
                    if (not args.is_testing and len(successes) == successes.maxlen
                            and avg_success > cfg.success_threshold and num_obj < cfg.max_objects):
                        num_obj += 1
                        successes.clear()
                        scores.clear()
                        stage_iteration = 0  # restart exploration annealing for the new stage
                        logger.scalar('curriculum/num_obj', num_obj, trainer.iteration)
                        print('Curriculum: now using %d objects' % num_obj)
            obs = list(next_obs)
            if args.max_iterations and trainer.iteration >= args.max_iterations:
                break
            if finished:
                for i, o in zip(finished, venv.reset(num_obj, indices=finished)):
                    obs[i] = attach_initial(trainer, o)
    finally:
        venv.close()
        logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Learn to rearrange objects in a confined cabinet with pushing and '
                                                 'pick-and-place (double deep Q-learning).')
    parser.add_argument('--random_seed', type=int, default=13953)
    parser.add_argument('--is_testing', action='store_true', default=False)
    parser.add_argument('--load_snapshot', action='store_true', default=False, help='load pre-trained weights')
    parser.add_argument('--snapshot', default='logs/train/models/snapshot.pth')
    parser.add_argument('--snapshot_target', default=None, help='target network weights (defaults to --snapshot)')
    parser.add_argument('--logging_directory', default='train')
    parser.add_argument('--save_visualizations', action='store_true', default=False)
    parser.add_argument('--save_every', type=int, default=500, help='environment steps between model snapshots')
    parser.add_argument('--num_obj', type=int, default=1, help='initial number of objects (curriculum start)')
    parser.add_argument('--max_obj', type=int, default=8, help='maximum number of objects of the curriculum')
    parser.add_argument('--max_actions', type=int, default=40, help='maximum number of actions per episode')
    parser.add_argument('--num_envs', type=int, default=1, help='parallel simulators')
    parser.add_argument('--updates_per_step', type=int, default=1, help='gradient steps per environment step')
    SimParams.add_arguments(parser)
    parser.add_argument('--eps_decay', type=float, default=0.9995,
                        help='epsilon multiplier per environment step (0.8 -> 0.01); use ~0.998 for runs of a few thousand steps')
    parser.add_argument('--placement', choices=['heuristic', 'learned'], default='heuristic',
                        help="'learned': a Q-head chooses where to place the held object (default: rule from the paper)")
    parser.add_argument('--belief', choices=['none', 'spatial', 'recurrent'], default='none',
                        help='belief state ablation: memoryless / spatial memory map / ConvGRU latent')
    parser.add_argument('--randomize', action='store_true', default=False, help='domain randomisation')
    parser.add_argument('--max_iterations', type=int, default=0, help='stop after this many environment steps (0 = run forever)')
    parser.add_argument('--start_method', default='spawn', help='multiprocessing start method of the workers')
    parser.add_argument('--gui', action='store_true', default=False, help='show the PyBullet GUI (first env only)')
    main(parser.parse_args())
