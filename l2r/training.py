"""Training loop: double DQN on push + pick-and-place with a curriculum, hindsight replay and parallel simulators."""
import argparse
import os
from collections import deque
from functools import partial

import cv2
import numpy as np

from l2r.config import Config, SimParams
from l2r.envs.rearrange import make_env
from l2r.envs.vec import VecEnv
from l2r.learning.memory import Transition
from l2r.learning.policy import attach_initial, carry, carry_over, select_for
from l2r.learning.factory import make_trainer
from l2r.logger import Logger
from l2r.sim.workspace import Workspace
from l2r.utils import seed_everything


class Curriculum:
    """Number of objects per episode: start small, add one when the recent success rate is high (paper, IV-D)."""

    def __init__(self, cfg, frozen=False, validated=False):
        """frozen: never advance (testing). validated: do not advance on the training window; the training loop calls
        `advance` after a greedy validation instead."""
        self.cfg, self.frozen, self.validated = cfg, frozen, validated
        self.num_obj = cfg.start_objects
        self.successes = deque(maxlen=cfg.success_window)
        self.returns = deque(maxlen=cfg.success_window)
        self.success_rate = self.mean_return = 0.0  # over the window, as of the last recorded episode

    def record(self, success, episode_return):
        """Log a finished episode; returns True if the number of objects was increased."""
        self.successes.append(1 if success else 0)
        self.returns.append(episode_return)
        self.success_rate = sum(self.successes) / len(self.successes)
        self.mean_return = sum(self.returns) / len(self.returns)
        full = len(self.successes) == self.successes.maxlen
        if (not self.frozen and not self.validated and full and self.success_rate > self.cfg.success_threshold):
            return self.advance()
        return False

    def advance(self):
        """One more object (up to the maximum); returns True if the number changed."""
        if self.frozen or self.num_obj >= self.cfg.max_objects:
            return False
        self.num_obj += 1
        self.successes.clear()
        self.returns.clear()
        return True


class TrainingLoop:
    """Owns the parallel simulators, the learner and the logging; `run` steps all simulators in lock step."""

    def __init__(self, args):
        self.args = args
        self.cfg = Config.from_args(args)
        params = SimParams.from_args(args)
        self.ws = Workspace.from_params(params)  # geometry for the main process (the workers build their own)
        seed_everything(args.random_seed)
        self.log_dir = os.path.join(os.path.abspath('logs'), args.logging_directory)
        os.makedirs(os.path.join(self.log_dir, 'models'), exist_ok=True)
        n = args.num_envs
        self.venv = VecEnv([partial(make_env, self.cfg, args.gui and r == 0, self.cfg.start_objects, params,
                                    args.random_seed + r) for r in range(n)], start_method=args.start_method)
        self.trainer = make_trainer(self.cfg, is_testing=args.is_testing, load_snapshot=args.load_snapshot,
                               snapshot=args.snapshot, snapshot_target=args.snapshot_target)
        self.logger = Logger(self.log_dir)
        if args.demos:
            from l2r.demos import adapt_demos, load_demos
            demos, stats = load_demos(args.demos)
            demos = adapt_demos(demos, self.cfg)
            print('Demonstrations: %d transitions (expert success %.2f over %d episodes)'
                  % (len(demos), stats['success_rate'], stats['episodes']))
            self.trainer.demos = demos
            if args.pretrain_steps:
                self.trainer.pretrain(demos, args.pretrain_steps)
        self.curriculum = Curriculum(self.cfg, frozen=args.is_testing, validated=bool(args.curriculum_eval))
        self.validations = 0
        self.episode_objects = [self.curriculum.num_obj] * args.num_envs  # objects of each simulator's running episode
        self.stage_iteration = 0  # steps since the last curriculum change (epsilon is annealed per stage)
        self.episode = 0

    @property
    def epsilon(self):
        """epsilon-greedy: 0.8 annealed to 0.01 for each curriculum stage (paper)."""
        if self.args.is_testing:
            return 0.0
        return max(self.cfg.eps_start * self.cfg.eps_decay ** self.stage_iteration, self.cfg.eps_end)

    def run(self):
        args, n = self.args, self.args.num_envs
        seeds = [args.random_seed + 1000 * r for r in range(n)]
        self.episode_objects = self._draw_objects(n, mix=False)
        obs = [attach_initial(self.trainer, o) for o in self.venv.reset(self.episode_objects, seeds=seeds)]
        try:
            while True:
                eps = self.epsilon
                q = self.trainer.forward_batch(obs)
                choices = [select_for(self.trainer, obs[i], q, i, eps, self.cfg, self.ws) for i in range(n)]
                next_obs, rewards, terminated, truncated, infos = self.venv.step(
                    [{'primitive': c[0], 'pix': c[1]} for c in choices])
                for i in range(n):
                    carry(self.trainer, next_obs[i], None if q.hidden is None else q.hidden[i], choices[i][0],
                          choices[i][1], infos[i]['place_pix'])
                if args.save_visualizations:
                    self._save_visualization(q, obs[0], choices[0], infos[0])
                if not args.is_testing:
                    self._learn(obs, next_obs, rewards, terminated, choices, infos, eps)
                self.trainer.iteration += n
                self.stage_iteration += n
                finished = [i for i in range(n) if terminated[i] or truncated[i]]
                for i in finished:
                    self._finish_episode(infos[i], eps, self.episode_objects[i])
                obs = list(next_obs)
                if self._validation_due(finished):
                    obs = self._validate_and_reset(n)
                elif finished:
                    counts = self._draw_objects(len(finished))
                    for i, count, o in zip(finished, counts, self.venv.reset(counts, indices=finished)):
                        obs[i] = attach_initial(self.trainer, o)
                        self.episode_objects[i] = count
                if args.max_iterations and self.trainer.iteration >= args.max_iterations:
                    break
                if args.max_episodes and self.episode >= args.max_episodes:
                    break
        finally:
            self.venv.close()
            self.logger.close()

    # ------------------------------------------------------------------ learning
    def _learn(self, obs, next_obs, rewards, terminated, choices, infos, eps):
        """Store the transitions (and their hindsight relabelling), learn from them, snapshot periodically."""
        trainer, args, n = self.trainer, self.args, self.args.num_envs
        batch = []
        for i in range(n):
            primitive, pix, _ = choices[i]
            transition = Transition(obs[i], pix, next_obs[i], rewards[i], primitive, terminated[i])
            trainer.buffer.push(*transition)
            batch.append(transition)
            if infos[i]['relabel']:  # hindsight experience replay (paper Fig. 8)
                obs_r, next_r = infos[i]['relabel_pair']
                carry_over(trainer, obs[i], obs_r)
                carry_over(trainer, next_obs[i], next_r)
                trainer.buffer.push(obs_r, pix, next_r, self.cfg.reward_goal, primitive, True)
        trainer.train_step(batch)  # short-term memory: learn from the newest transitions at once
        for _ in range(args.updates_per_step * n):
            loss = trainer.optimize()
            if loss is not None and trainer.opt_steps % 10 == 0:
                self._log_training(loss, eps)
        if trainer.iteration // args.save_every != (trainer.iteration - n) // args.save_every:
            trainer.save(os.path.join(self.log_dir, 'models', 'snapshot-%07d.pth' % trainer.iteration),
                         os.path.join(self.log_dir, 'models', 'target-%07d.pth' % trainer.iteration))

    def _log_training(self, loss, eps):
        t, step = self.trainer, self.trainer.iteration
        self.logger.scalar('train/loss', loss, step)
        self.logger.scalar('train/q_mean', t.last_q, step)
        self.logger.scalar('train/epsilon', eps, step)
        self.logger.scalar('train/lr', t.optimizer.param_groups[0]['lr'], step)
        self.logger.scalar('train/buffer', len(t.buffer), step)
        print('Iteration %d  loss %.4f  eps %.3f' % (step, loss, eps))

    def _draw_objects(self, k, mix=True):
        """Object counts of k new episodes: the current stage, or (probability --stage_mix) a uniformly drawn earlier
        stage, so the replay buffer keeps states of the easier stages and forgetting shows up in training."""
        current = self.curriculum.num_obj
        counts = [current] * k
        if mix and self.args.stage_mix > 0 and current > self.cfg.start_objects:
            counts = [int(np.random.randint(self.cfg.start_objects, current)) if np.random.rand() < self.args.stage_mix
                      else current for _ in range(k)]
        return counts

    def _finish_episode(self, info, eps, objects):
        self.episode += 1
        advanced = self.curriculum.record(info['sorted'], info['episode_return'])
        c = self.curriculum
        print('Episode %d: return %.0f, success %s, avg success %.2f, avg return %.1f'
              % (self.episode, info['episode_return'], info['sorted'], c.success_rate, c.mean_return))
        self.logger.episode(self.episode, self.trainer.iteration, objects, info['episode_return'],
                            info['actions'], info['sorted'], eps)
        if advanced:
            self.stage_iteration = 0  # restart exploration annealing for the new stage
            self.logger.scalar('curriculum/num_obj', c.num_obj, self.trainer.iteration)
            print('Curriculum: now using %d objects' % c.num_obj)

    def _validation_due(self, finished):
        a = self.args
        return bool(a.curriculum_eval and not a.is_testing and finished and self.episode % a.curriculum_eval_every < len(finished)
                    and self.episode >= a.curriculum_eval_every)

    def _validate_and_reset(self, n):
        """Greedy (epsilon = 0) validation on fresh random configurations at the current object count; advance the
        curriculum if the success rate reaches the threshold. All simulators are reset afterwards."""
        from l2r.evaluation import evaluate
        c, a = self.curriculum, self.args
        seed = 10 ** 6 + self.validations * 1000
        result = evaluate(self.trainer, self.venv, self.cfg, self.ws, c.num_obj, a.curriculum_eval, seed)
        self.validations += 1
        rate = result['success_rate']
        passed = rate >= self.cfg.success_threshold
        print('Validation %d (%d objects, %d greedy episodes): success %.2f, mean actions %s -> %s'
              % (self.validations, c.num_obj, result['episodes'], rate, result['mean_actions_success'],
                 'advance' if passed else 'stay'))
        self.logger.scalar('curriculum/val_success', rate, self.trainer.iteration)
        with open(os.path.join(self.log_dir, 'validation.csv'), 'a') as f:
            f.write('%d,%d,%d,%.3f\n' % (self.validations, self.trainer.iteration, c.num_obj, rate))
        if passed and c.advance():
            self.stage_iteration = 0
            self.logger.scalar('curriculum/num_obj', c.num_obj, self.trainer.iteration)
            print('Curriculum: now using %d objects' % c.num_obj)
        self.episode_objects = [c.num_obj] * self.args.num_envs
        return [attach_initial(self.trainer, o) for o in self.venv.reset(self.episode_objects)]

    def _save_visualization(self, q, obs, choice, info):
        primitive, pix, _ = choice
        maps = {'push': q.push, 'grasp': q.pick, 'place': q.place}[primitive][0]
        if primitive == 'push':  # the two directions of the chosen length
            n_len = len(self.cfg.push_lengths)
            maps, pix = maps[[pix[0] % n_len, n_len + pix[0] % n_len]], (pix[0] // n_len,) + tuple(pix[1:])
        place = info['place_pix']
        image = self.trainer.get_prediction_vis(maps, obs['colormap'], pix, pix if place is None else [place[1], place[0]])
        cv2.imwrite(os.path.join(self.log_dir, 'visualization.%s.png' % primitive), image)


def add_arguments(parser):
    """Command-line flags of training (shared with `evaluate` where noted there)."""
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
    parser.add_argument('--eps_decay', type=float, default=0.9995,
                        help='epsilon multiplier per environment step (0.8 -> 0.01); ~0.998 for runs of a few thousand steps')
    Config.add_arguments(parser)
    SimParams.add_arguments(parser)
    parser.add_argument('--randomize', action='store_true', default=False, help='domain randomisation')
    parser.add_argument('--stage_mix', type=float, default=0.0,
                        help='probability that a new training episode uses a uniformly drawn EARLIER curriculum stage '
                             '(keeps easier states in the replay buffer; 0 = current stage only)')
    parser.add_argument('--curriculum_eval', type=int, default=0,
                        help='advance the curriculum after a greedy validation on this many fresh random configurations '
                             '(success >= threshold) instead of the epsilon-greedy training window (0 = training window)')
    parser.add_argument('--curriculum_eval_every', type=int, default=25, help='training episodes between validations')
    parser.add_argument('--demos', default='', help='recorded expert demonstrations (scripts/collect_demos.py; same model flags)')
    parser.add_argument('--pretrain_steps', type=int, default=0, help='gradient steps on the demonstrations before training (DQfD)')
    parser.add_argument('--max_episodes', type=int, default=0, help='stop after this many finished episodes (0 = no cap)')
    parser.add_argument('--max_iterations', type=int, default=0, help='stop after this many environment steps (0 = run forever)')
    parser.add_argument('--start_method', default='spawn', help='multiprocessing start method of the workers')
    parser.add_argument('--gui', action='store_true', default=False, help='show the PyBullet GUI (first env only)')


def run(args):
    TrainingLoop(args).run()


def main(argv=None):
    parser = argparse.ArgumentParser(description='Learn to rearrange objects in a confined cabinet with pushing and '
                                                 'pick-and-place (double deep Q-learning).')
    add_arguments(parser)
    run(parser.parse_args(argv))
