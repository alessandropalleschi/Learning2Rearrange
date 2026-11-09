"""Record scripted-expert demonstrations: `python scripts/collect_demos.py --help`."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from l2r.config import Config, SimParams  # noqa: E402
from l2r.demos import record_episodes, save_demos  # noqa: E402
from l2r.envs.rearrange import RearrangeEnv  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description='Record scripted-expert demonstrations.')
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--objects', type=int, nargs=2, default=[1, 5], metavar=('MIN', 'MAX'))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', default='data/demos/expert.pkl')
    Config.add_arguments(parser)
    SimParams.add_arguments(parser)
    parser.add_argument('--randomize', action='store_true', default=False)
    args = parser.parse_args(argv)
    cfg = Config.from_args(args)
    cfg.max_actions = 60
    np.random.seed(args.seed)
    env = RearrangeEnv(cfg, num_obj=args.objects[0], params=SimParams.from_args(args))
    transitions, stats = record_episodes(env, args.episodes, args.objects, args.seed)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    save_demos(args.out, transitions, stats)
    print('saved %s: %s' % (args.out, stats))


if __name__ == '__main__':
    main()
