"""DAgger with the scripted expert: `python scripts/dagger.py --variant object --demos data/demos/expert3.pkl`."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from l2r.arch_benchmark import VARIANTS, make_env, make_variant, rollout  # noqa: E402
from l2r.dagger import dagger  # noqa: E402
from l2r.demos import load_demos  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description='DAgger with the scripted expert.')
    parser.add_argument('--demos', default='data/demos/expert.pkl')
    parser.add_argument('--variant', default='object', choices=sorted(VARIANTS))
    parser.add_argument('--rounds', type=int, default=6)
    parser.add_argument('--episodes', type=int, default=20, help='learner episodes per round')
    parser.add_argument('--steps', type=int, default=1500, help='gradient steps per round')
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--beta', type=float, default=0.5, help='expert mixture of the first round (halved each round, 0 in the last)')
    parser.add_argument('--objects', type=int, nargs=2, default=[1, 4])
    parser.add_argument('--final_rollouts', type=int, default=50, help='held-out greedy episodes per object count (after BC and after DAgger)')
    parser.add_argument('--eval_objects', type=int, nargs='+', default=[4, 5], help='object counts of the held-out evaluation')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', default='')
    args = parser.parse_args(argv)
    demos, _ = load_demos(args.demos)
    torch.manual_seed(args.seed)
    variant = make_variant(args.variant, (args.rounds + 1) * args.steps, args.seed, args.batch)
    env = make_env(variant.cfg, args.objects[0], max_actions=20)
    def evaluate(stage):  # the same held-out seeds (7000 + episode) for every variant and stage
        result = {n: rollout(variant, args.final_rollouts, n, env=env) for n in args.eval_objects}
        print('%s held-out rollouts: %s' % (stage, result), flush=True)
        return result

    results = {}
    history = dagger(variant, env, demos, args.rounds, args.episodes, args.steps, args.batch, tuple(args.objects), args.beta, args.seed,
                     after_pretrain=lambda: results.__setitem__('bc', evaluate('BC only')))
    results['dagger'] = evaluate('DAgger')
    if args.out:
        with open(args.out, 'w') as f:
            json.dump({'args': vars(args), 'history': history, 'results': results}, f, indent=1)


if __name__ == '__main__':
    main()
