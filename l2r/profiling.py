"""Where does the simulation time go, and how do the simulator options change the behaviour?

    python profile_sim.py --actions 100 --seed 1                              # defaults
    python profile_sim.py --actions 100 --seed 1 --sim_quality modern         # compare option sets on the same seed
    python profile_sim.py --actions 20 --step_distance 0.004 --move_budget 400

Runs random actions in one simulator. Time: stepSimulation = physics, calculateInverseKinematics = IK,
getCameraImage = rendering. Behaviour: attempted / executed actions, force-limit stops (and where they happen),
sorted episodes, occluded fraction of the heightmap.
"""
import argparse
import cProfile
import pstats
import time

import numpy as np

from l2r.config import Config, SimParams
from l2r.envs.rearrange import RearrangeEnv
from l2r.learning.policy import select_for
from l2r.learning.trainer import Trainer
from l2r.sim.constants import OCCLUDED_RGB
from l2r.sim.workspace import Workspace

PRIMITIVES = ('push', 'grasp', 'place')


def run_random_actions(env, trainer, cfg, ws, n_actions):
    """Execute `n_actions` random actions; returns the measurements as a dict."""
    stats = {'times': {k: [] for k in PRIMITIVES}, 'attempted': dict.fromkeys(PRIMITIVES, 0),
             'executed': dict.fromkeys(PRIMITIVES, 0), 'stops': {k: [] for k in PRIMITIVES}, 'occluded': [],
             'policy_times': [], 'force_stops': 0, 'episodes': 0, 'sorted': 0}
    obs = env.reset()
    for _ in range(n_actions):
        t0 = time.time()
        q = trainer.forward_batch([obs])
        primitive, pix, _ = select_for(trainer, obs, q, 0, 1.0, cfg, ws)
        t1 = time.time()
        obs, _, terminated, truncated, info = env.step({'primitive': primitive, 'pix': pix})
        stats['times'][primitive].append(time.time() - t1)
        stats['policy_times'].append(t1 - t0)
        stats['executed'][primitive] += int(info['executed'])
        stats['attempted'][primitive] += int(info['attempted'])
        stats['force_stops'] += info.get('force_stops', 0)
        stats['stops'][primitive] += env.sim.force_events
        env.sim.force_events.clear()
        stats['occluded'].append(float(np.mean(np.all(obs['colormap'] == OCCLUDED_RGB, axis=-1))))
        if terminated or truncated:
            stats['episodes'] += 1
            stats['sorted'] += int(info['sorted'])
            obs = env.reset()
    return stats


def print_report(stats, n_actions, ws):
    times = stats['times']
    print('\n== time per action (s) ==')
    for k in PRIMITIVES:
        if times[k]:
            print('%-6s n=%2d  mean %.2f s  max %.2f s  attempted %d  executed %d  force-stops %d'
                  % (k, len(times[k]), np.mean(times[k]), np.max(times[k]), stats['attempted'][k], stats['executed'][k],
                     len(stats['stops'][k])))
    print('policy forward + action selection: mean %.3f s' % np.mean(stats['policy_times']))
    print('simulated actions/s: %.2f' % (n_actions / max(sum(sum(v) for v in times.values()), 1e-9)))
    print('motions stopped by the force limit: %d   episodes sorted: %d/%d'
          % (stats['force_stops'], stats['sorted'], stats['episodes']))
    print('mean occluded (grey) heightmap fraction: %.3f' % np.mean(stats['occluded']))
    stops = [s for v in stats['stops'].values() for s in v]
    if stops:
        x, force = np.array(stops).T
        front = ws.bounds[0][1]
        print('force stops: %d outside the cabinet front (x > %.2f, free space!), %d inside; force median %.0f N, p90 %.0f N'
              % ((x > front).sum(), front, (x <= front).sum(), np.median(force), np.percentile(force, 90)))


def main(argv=None):
    parser = argparse.ArgumentParser(description='Profile the simulator.')
    parser.add_argument('--actions', type=int, default=20)
    parser.add_argument('--num_obj', type=int, default=3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--top', type=int, default=15, help='functions to list in the profile')
    Config.add_arguments(parser)
    SimParams.add_arguments(parser)
    parser.add_argument('--randomize', action='store_true', default=False, help='domain randomisation')
    args = parser.parse_args(argv)

    cfg = Config.from_args(args)
    cfg.max_actions = 10 ** 6
    params = SimParams.from_args(args)
    ws = Workspace.from_params(params)
    np.random.seed(args.seed)
    env = RearrangeEnv(cfg, num_obj=args.num_obj, params=params)
    trainer = Trainer(cfg, is_testing=True)
    profiler = cProfile.Profile()
    profiler.enable()
    stats = run_random_actions(env, trainer, cfg, ws, args.actions)
    profiler.disable()
    print_report(stats, args.actions, ws)
    print('\n== top functions by own time ==')
    pstats.Stats(profiler).sort_stats('tottime').print_stats(args.top)
