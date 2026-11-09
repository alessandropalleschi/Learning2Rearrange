"""Training / testing loop: double DQN with push + pick-and-place primitives, curriculum and HER."""
import argparse
import csv
import os
from collections import deque

import cv2
import numpy as np

from config import Config
from memory import Transition
from rearrange_env import RearrangeEnv, check_push
from trainer import Trainer


def select_action(trainer, env, obs, push_q, pick_q, explore_prob, is_testing, cfg):
    """Epsilon-greedy over the two primitives. Returns (primitive, (rot,row,col), predicted_value)."""
    push_q = trainer.preprocessQmaps(push_q, obs['depthmap'], 'push')
    pick_q = trainer.preprocessQmaps(pick_q, obs['depthmap'], 'grasp')
    primitive = 'push' if np.max(push_q) > np.max(pick_q) else 'grasp'
    q = {'push': push_q, 'grasp': pick_q}

    if not is_testing and np.random.uniform() < explore_prob:
        if np.random.randint(0, 2) == 1:
            primitive = 'push' if np.random.randint(0, 2) == 0 else 'grasp'
        valid = np.flatnonzero(q[primitive].ravel() > -1000)
        if valid.size:
            if primitive == 'push':  # prefer pushes that actually touch something
                for _ in range(100):
                    pix = np.unravel_index(np.random.choice(valid), q[primitive].shape)
                    angle = np.deg2rad(pix[0] * 180.0 - 90)
                    if check_push(env.sim, obs['depthmap'], pix[2], pix[1], angle, cfg.push_length):
                        break
            else:
                pix = np.unravel_index(np.random.choice(valid), q[primitive].shape)
        else:
            pix = np.unravel_index(np.argmax(q[primitive]), q[primitive].shape)
    else:
        pix = np.unravel_index(np.argmax(q[primitive]), q[primitive].shape)
    return primitive, tuple(int(i) for i in pix), float(q[primitive][pix])


def main(args):
    cfg = Config(start_objects=args.num_obj, max_objects=args.max_obj, max_actions=args.max_actions)
    np.random.seed(args.random_seed)
    log_dir = os.path.join(os.path.abspath('logs'), args.logging_directory)
    os.makedirs(os.path.join(log_dir, 'models'), exist_ok=True)

    env = RearrangeEnv(cfg, gui=args.gui, num_obj=cfg.start_objects)
    trainer = Trainer(cfg, is_testing=args.is_testing, load_snapshot=args.load_snapshot,
                      snapshot=args.snapshot, snapshot_target=args.snapshot_target)

    num_obj = cfg.start_objects
    successes = deque(maxlen=cfg.success_window)
    scores = deque(maxlen=cfg.success_window)
    stage_iteration, episode = 0, 0
    csv_file = open(os.path.join(log_dir, 'episodes.csv'), 'a', newline='')
    writer = csv.writer(csv_file)
    if csv_file.tell() == 0:
        writer.writerow(['episode', 'iteration', 'num_obj', 'return', 'actions', 'success', 'epsilon'])

    obs = env.reset(num_obj)
    while True:
        # epsilon-greedy: 0.8 annealed to 0.01 for each curriculum stage (paper)
        eps = max(cfg.eps_start * cfg.eps_decay ** stage_iteration, cfg.eps_end)

        push_q, pick_q = trainer.forward(obs['colormap'], obs['depthmap'])
        primitive, pix, value = select_action(trainer, env, obs, push_q, pick_q, eps, args.is_testing, cfg)
        next_obs, reward, terminated, truncated, info = env.step({'primitive': primitive, 'pix': pix})

        if args.save_visualizations:
            vis = trainer.get_prediction_vis(push_q if primitive == 'push' else pick_q, obs['colormap'], pix,
                                             pix if info['place_pix'] is None else [info['place_pix'][1], info['place_pix'][0]])
            cv2.imwrite(os.path.join(log_dir, 'visualization.%s.png' % primitive), vis)

        if not args.is_testing:
            transition = Transition(obs, pix, next_obs, reward, primitive, terminated)
            trainer.buffer.push(*transition)
            trainer.train_step([transition])  # short-term memory: learn from the newest transition at once
            if info['relabel']:
                obs_r, next_r = env.relabel(obs, next_obs)
                print('Relabeling')
                trainer.buffer.push(obs_r, pix, next_r, cfg.reward_goal, primitive, True)
            loss = trainer.optimize()
            if loss is not None:
                print('Iteration %d  loss %.4f  eps %.3f' % (trainer.iteration, loss, eps))
            if trainer.iteration % args.save_every == 0:
                trainer.save(os.path.join(log_dir, 'models', 'snapshot-%06d.pth' % trainer.iteration),
                             os.path.join(log_dir, 'models', 'target-%06d.pth' % trainer.iteration))
        trainer.iteration += 1
        stage_iteration += 1

        if terminated or truncated:
            episode += 1
            successes.append(1 if info['sorted'] else 0)
            scores.append(info['episode_return'])
            avg_success = sum(successes) / len(successes)
            print('Episode %d: return %.0f, success %s, avg success %.2f, avg return %.1f'
                  % (episode, info['episode_return'], info['sorted'], avg_success, sum(scores) / len(scores)))
            writer.writerow([episode, trainer.iteration, num_obj, info['episode_return'], info['actions'],
                             int(info['sorted']), round(eps, 4)])
            csv_file.flush()
            if (not args.is_testing and len(successes) == successes.maxlen and avg_success > cfg.success_threshold
                    and num_obj < cfg.max_objects):
                num_obj += 1
                successes.clear()
                scores.clear()
                stage_iteration = 0  # restart exploration annealing for the new stage
                print('Curriculum: now using %d objects' % num_obj)
            obs = env.reset(num_obj)
        else:
            obs = next_obs


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
    parser.add_argument('--save_every', type=int, default=500, help='iterations between model snapshots')
    parser.add_argument('--num_obj', type=int, default=1, help='initial number of objects (curriculum start)')
    parser.add_argument('--max_obj', type=int, default=8, help='maximum number of objects of the curriculum')
    parser.add_argument('--max_actions', type=int, default=40, help='maximum number of actions per episode')
    parser.add_argument('--gui', action='store_true', default=False, help='show the PyBullet GUI')
    main(parser.parse_args())
