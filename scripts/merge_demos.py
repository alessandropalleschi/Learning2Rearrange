"""Merge demonstration parts: `python scripts/merge_demos.py data/demos/parts2 data/demos/expert2.pkl`."""
import glob
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from l2r.demos import save_demos  # noqa: E402
from l2r.learning.memory import Transition  # noqa: E402

parts, out = sys.argv[1], sys.argv[2]
transitions, episodes, success, actions = [], 0, 0.0, 0.0
for path in sorted(glob.glob(os.path.join(parts, 'p*.pkl'))):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    transitions += [Transition(*t) for t in data['transitions']]
    n = data['stats']['episodes']
    episodes += n
    success += data['stats']['success_rate'] * n
    actions += data['stats']['mean_actions'] * n
stats = {'episodes': episodes, 'success_rate': success / episodes, 'mean_actions': actions / episodes, 'transitions': len(transitions)}
save_demos(out, transitions, stats)
print(out, stats)
