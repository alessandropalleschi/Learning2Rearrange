"""Train and evaluate the policies of the ablation study, then summarise them.

    python scripts/run_experiments.py list
    python scripts/run_experiments.py run --suite core --seeds 0 1 2 --steps 20000 --out runs/core
    python scripts/run_experiments.py run --suite smoke --out runs/smoke           # a few minutes: checks the whole chain
    python scripts/run_experiments.py summarize --out runs/core                    # table (markdown + csv) with mean +- std

Every experiment is one `train.py` run followed by one `evaluate.py` run per seed; finished runs are skipped (resume-safe).
A suite is a list of named experiments = flag sets; the first one of a suite is the baseline the others are compared with.
Results of one run: <out>/<experiment>/seed<k>/{train.log, eval.log, eval.json, models/}.
"""
import argparse
import csv
import json
import os
import shlex
import statistics
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- named experiments (flags of train.py / evaluate.py; the evaluation must use the same architecture flags) ---
BASE = ['--placement', 'learned', '--footprint_placement']
EXPERIMENTS = {
    # baseline: dense Q-maps, GroupNorm, global context, shaped reward (the default), learned placement
    'baseline': BASE,
    'sparse_reward': BASE + ['--reward_mode', 'sparse'],               # the paper's reward
    'reward_distance': BASE + ['--reward_mode', 'distance'],
    'reward_noop': BASE + ['--reward_mode', 'noop'],
    'push_lengths': BASE + ['--push_lengths', '0.04', '0.08', '0.12'],
    'hierarchy': BASE + ['--hierarchy'],
    'hybrid_transformer': BASE + ['--backbone', 'hybrid'],
    'belief_spatial': BASE + ['--belief', 'spatial'],
    'belief_recurrent': BASE + ['--belief', 'recurrent'],
    'heuristic_placement': ['--placement', 'heuristic'],
    'modern_physics': BASE + ['--physics', 'modern', '--sim_quality', 'modern'],
    'perception_network': BASE + ['--perception', 'network'],
    'perception_agnostic': BASE + ['--perception', 'agnostic', '--perception_weights', 'assets/perception/agnostic.pth'],
    'gripper': BASE + ['--end_effector', 'gripper', '--jaw_yaws', '5'],
    'gripper_hierarchy': BASE + ['--end_effector', 'gripper', '--hierarchy'],
}
SUITES = {
    'smoke': ['baseline', 'hierarchy'],
    'core': ['baseline', 'sparse_reward', 'reward_distance', 'push_lengths', 'hierarchy', 'hybrid_transformer', 'heuristic_placement'],
    'belief': ['baseline', 'belief_spatial', 'belief_recurrent'],
    'perception': ['baseline', 'perception_network', 'perception_agnostic'],
    'end_effector': ['baseline', 'gripper', 'gripper_hierarchy'],
    'physics': ['baseline', 'modern_physics'],
    'all': list(EXPERIMENTS),
}
METRICS = ['success_rate', 'mean_actions_success', 'mean_return']  # keys of eval.json (one entry per object count)


def run(cmd, log):
    print('$', ' '.join(shlex.quote(c) for c in cmd), flush=True)
    with open(log, 'w') as f:
        return subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode


def run_one(name, flags, seed, args):
    directory = os.path.abspath(os.path.join(args.out, name, f'seed{seed}'))
    os.makedirs(directory, exist_ok=True)
    result = os.path.join(directory, 'eval.json')
    if os.path.exists(result) and not args.force:
        print('skip (done):', name, 'seed', seed)
        return result
    logs = os.path.relpath(directory, os.path.join(ROOT, 'logs'))  # train.py / evaluate.py write below logs/
    common = [sys.executable]
    train = common + ['train.py', '--logging_directory', logs, '--random_seed', str(seed), '--num_envs', str(args.num_envs),
                      '--max_obj', str(args.max_obj), '--num_obj', str(args.num_obj), '--max_iterations', str(args.steps),
                      '--max_episodes', str(args.max_episodes),
                      '--save_every', str(max(args.steps // 4, 100)), *flags, *args.extra]
    if run(train, os.path.join(directory, 'train.log')):
        raise SystemExit(f'training failed: see {directory}/train.log')
    models = os.path.join(ROOT, 'logs', logs, 'models')
    candidates = sorted(f for f in os.listdir(models) if f.startswith('snapshot') and f.endswith('.pth') and 'target' not in f)
    final = os.path.join(models, 'snapshot.pth') if os.path.exists(os.path.join(models, 'snapshot.pth')) else os.path.join(models, candidates[-1])
    evaluate = common + ['evaluate.py', '--snapshot', final, '--objects', *map(str, args.objects), '--episodes', str(args.episodes),
                         '--num_envs', str(args.num_envs), '--seed', str(1000 + seed), '--logging_directory', logs + '_eval', *flags]
    if run(evaluate, os.path.join(directory, 'eval.log')):
        raise SystemExit(f'evaluation failed: see {directory}/eval.log')
    produced = os.path.join(ROOT, 'logs', logs + '_eval', 'eval.json')
    with open(produced) as f, open(result, 'w') as g:
        g.write(f.read())
    return result


def load(result):
    with open(result) as f:
        rows = json.load(f)
    return rows if isinstance(rows, list) else [rows]


def summarize(args):
    table = {}
    for name in sorted(os.listdir(args.out)):
        for seed_dir in sorted(os.listdir(os.path.join(args.out, name))):
            path = os.path.join(args.out, name, seed_dir, 'eval.json')
            if os.path.exists(path):
                for row in load(path):
                    table.setdefault(name, {}).setdefault(row.get('num_obj', row.get('objects', '?')), []).append(row)
    keys = sorted({k for rows in table.values() for group in rows.values() for r in group for k, v in r.items()
                   if isinstance(v, (int, float)) and k not in ('num_obj', 'objects', 'episodes')})
    keys = [k for k in METRICS if k in keys] + [k for k in keys if k not in METRICS][:3]
    lines = ['| experiment | objects | seeds | ' + ' | '.join(keys) + ' |', '|' + '---|' * (3 + len(keys))]
    csv_rows = [['experiment', 'objects', 'seeds'] + keys]
    for name, groups in table.items():
        for objects, group in sorted(groups.items(), key=lambda kv: str(kv[0])):
            cells = []
            for k in keys:
                values = [r[k] for r in group if k in r]
                cells.append('%.3f +- %.3f' % (statistics.mean(values), statistics.pstdev(values)) if values else '-')
            lines.append(f'| {name} | {objects} | {len(group)} | ' + ' | '.join(cells) + ' |')
            csv_rows.append([name, objects, len(group)] + cells)
    text = '\n'.join(lines)
    print(text)
    with open(os.path.join(args.out, 'summary.md'), 'w') as f:
        f.write(text + '\n')
    with open(os.path.join(args.out, 'summary.csv'), 'w', newline='') as f:
        csv.writer(f).writerows(csv_rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('list')
    r = sub.add_parser('run')
    r.add_argument('--suite', choices=sorted(SUITES), default=None)
    r.add_argument('--experiments', nargs='+', choices=sorted(EXPERIMENTS), default=None)
    r.add_argument('--seeds', type=int, nargs='+', default=[0])
    r.add_argument('--steps', type=int, default=20000, help='environment steps per training run')
    r.add_argument('--num_envs', type=int, default=4)
    r.add_argument('--max_obj', type=int, default=5, help='top of the curriculum')
    r.add_argument('--num_obj', type=int, default=1, help='start of the curriculum (2 objects are easy even for random actions)')
    r.add_argument('--max_episodes', type=int, default=0, help='cap on finished training episodes per run (0 = only --steps)')
    r.add_argument('--objects', type=int, nargs='+', default=[3, 4, 5], help='object counts of the evaluation')
    r.add_argument('--episodes', type=int, default=30, help='evaluation episodes per object count')
    r.add_argument('--out', default='runs/experiments')
    r.add_argument('--force', action='store_true')
    r.add_argument('--extra', nargs=argparse.REMAINDER, default=[], help='flags appended to every train.py call')
    s = sub.add_parser('summarize')
    s.add_argument('--out', default='runs/experiments')
    args = parser.parse_args(argv)
    if args.command == 'list':
        for suite, names in SUITES.items():
            print(f'{suite}: {", ".join(names)}')
        for name, flags in EXPERIMENTS.items():
            print(f'  {name:22s} {" ".join(flags)}')
    elif args.command == 'run':
        names = args.experiments or SUITES[args.suite or 'core']
        if args.suite == 'smoke':
            args.steps, args.episodes, args.objects, args.num_envs = min(args.steps, 400), 2, [3], 2
        for name in names:
            for seed in args.seeds:
                run_one(name, EXPERIMENTS[name], seed, args)
        summarize(args)
    else:
        summarize(args)


if __name__ == '__main__':
    main()
