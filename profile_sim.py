"""Where does the simulation time go?  Runs random actions in one simulator and prints a timing breakdown.

    python profile_sim.py --actions 20                       # defaults
    python profile_sim.py --actions 20 --step_distance 0.004  # compare a coarser straight-move step
    python profile_sim.py --actions 20 --move_budget 400      # compare a smaller joint-move budget

Look at the 'tottime' column: stepSimulation = physics, calculateInverseKinematics = IK,
getCameraImage = rendering, get_heightmap/get_observation = Python image processing.
"""
import argparse
import cProfile
import pstats
import time

import numpy as np

import env as sim_module
from config import Config, SimParams
from policy import select_action
from rearrange_env import RearrangeEnv
from trainer import Trainer


def main(args):
    cfg = Config(placement=args.placement, max_actions=10 ** 6)
    params = SimParams.preset(args.physics)
    params.step_distance, params.move_joints_max_steps = args.step_distance, args.move_budget
    sim_module.configure_cabinet(params)
    np.random.seed(args.seed)
    env = RearrangeEnv(cfg, num_obj=args.num_obj, params=params)
    trainer = Trainer(cfg, is_testing=True)
    obs = env.reset()
    times = {'push': [], 'grasp': [], 'place': []}
    obs_times = []
    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(args.actions):
        t0 = time.time()
        q = trainer.forward_batch([obs])
        prim, pix, _ = select_action(trainer, obs, q.push[0], q.pick[0], q.place[0], 1.0, cfg)  # random actions
        t1 = time.time()
        obs, reward, term, trunc, info = env.step({'primitive': prim, 'pix': pix})
        times[prim].append(time.time() - t1)
        obs_times.append(t1 - t0)
        if term or trunc:
            obs = env.reset()
    profiler.disable()
    print('\n== time per action (s) ==')
    for k, v in times.items():
        if v:
            print('%-6s n=%2d  mean %.2f  max %.2f' % (k, len(v), np.mean(v), np.max(v)))
    print('policy forward + action selection: mean %.3f s' % np.mean(obs_times))
    total = sum(sum(v) for v in times.values())
    print('simulated actions/s: %.2f' % (args.actions / max(total, 1e-9)))
    print('\n== top functions by own time ==')
    pstats.Stats(profiler).sort_stats('tottime').print_stats(args.top)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Profile the simulator.')
    parser.add_argument('--actions', type=int, default=20)
    parser.add_argument('--num_obj', type=int, default=3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--top', type=int, default=15)
    parser.add_argument('--step_distance', type=float, default=0.002)
    parser.add_argument('--move_budget', type=int, default=1000)
    parser.add_argument('--physics', choices=['legacy', 'paper'], default='legacy')
    parser.add_argument('--placement', choices=['heuristic', 'learned'], default='heuristic')
    main(parser.parse_args())
