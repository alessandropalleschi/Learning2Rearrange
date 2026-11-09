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

import l2r.sim.environment as sim_module
from l2r.config import Config, SimParams
from l2r.learning.policy import select_action
from l2r.envs.rearrange import RearrangeEnv
from l2r.learning.trainer import Trainer


def main(args):
    cfg = Config(placement=args.placement, max_actions=10 ** 6, push_lengths=tuple(args.push_lengths))
    params = SimParams.from_args(args)
    sim_module.configure_cabinet(params)
    np.random.seed(args.seed)
    env = RearrangeEnv(cfg, num_obj=args.num_obj, params=params)
    trainer = Trainer(cfg, is_testing=True)
    obs = env.reset()
    times = {'push': [], 'grasp': [], 'place': []}
    executed = {'push': 0, 'grasp': 0, 'place': 0}
    attempted = dict(executed)
    stops = {'push': [], 'grasp': [], 'place': []}  # (tool x, force) of every force-limit abort
    occluded = []  # fraction of grey (occluded) heightmap cells: shows sparse heightmaps at low camera scale
    force_stops = sorted_episodes = episodes = 0
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
        executed[prim] += int(info['executed'])
        attempted[prim] += int(info['attempted'])
        force_stops += info.get('force_stops', 0)
        stops[prim] += env.sim.force_events
        env.sim.force_events = []
        occluded.append(float(np.mean(np.all(obs['colormap'] == 105, axis=-1))))
        obs_times.append(t1 - t0)
        if term or trunc:
            episodes += 1
            sorted_episodes += int(info['sorted'])
            obs = env.reset()
    profiler.disable()
    print('\n== time per action (s) ==')
    for k, v in times.items():
        if v:
            print('%-6s n=%2d  mean %.2f s  max %.2f s  attempted %d  executed %d  force-stops %d'
                  % (k, len(v), np.mean(v), np.max(v), attempted[k], executed[k], len(stops[k])))
    print('policy forward + action selection: mean %.3f s' % np.mean(obs_times))
    total = sum(sum(v) for v in times.values())
    print('simulated actions/s: %.2f' % (args.actions / max(total, 1e-9)))
    print('motions stopped by the force limit: %d   episodes sorted: %d/%d' % (force_stops, sorted_episodes, episodes))
    print('mean occluded (grey) heightmap fraction: %.3f' % np.mean(occluded))
    all_stops = [s for v in stops.values() for s in v]
    if all_stops:
        x, f = np.array(all_stops).T
        front = sim_module.BOUNDS[0][1]
        print('force stops: %d outside the cabinet front (x > %.2f, free space!), %d inside; force median %.0f N, p90 %.0f N'
              % ((x > front).sum(), front, (x <= front).sum(), np.median(f), np.percentile(f, 90)))
    print('\n== top functions by own time ==')
    pstats.Stats(profiler).sort_stats('tottime').print_stats(args.top)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Profile the simulator.')
    parser.add_argument('--actions', type=int, default=20)
    parser.add_argument('--num_obj', type=int, default=3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--top', type=int, default=15)
    SimParams.add_arguments(parser)
    parser.add_argument('--push_lengths', type=float, nargs='+', default=[0.08])
    parser.add_argument('--placement', choices=['heuristic', 'learned'], default='heuristic')
    main(parser.parse_args())
