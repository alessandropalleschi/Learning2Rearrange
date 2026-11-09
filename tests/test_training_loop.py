"""Smoke test of the whole learning stack with a fake simulator (no PyBullet needed).

    python tests/test_training_loop.py
"""
import argparse
import os
import sys
import tempfile
import types

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
try:
    import pybullet  # noqa: F401
except ImportError:  # stub the simulator modules so the code can be imported
    for name in ['pybullet', 'pybullet_data', 'pybullet_utils', 'pybullet_utils.bullet_client']:
        sys.modules[name] = types.ModuleType(name)
    sys.modules['pybullet_utils'].bullet_client = sys.modules['pybullet_utils.bullet_client']

import rearrange_env  # noqa: E402
import train  # noqa: E402
import evaluate  # noqa: E402
from config import Config  # noqa: E402
from memory import PrioritizedReplay  # noqa: E402
from trainer import Trainer  # noqa: E402
from vec_env import VecEnv  # noqa: E402
from functools import partial  # noqa: E402

H, W = 100, 120


class FakeSim:
    """Minimal stand-in for env.Environment with the interface RearrangeEnv uses."""

    def __init__(self, gui=False, num_obj=(1,), params=None):
        self.num_obj, self.t, self.id_list, self.new_class = list(num_obj), 0, [1], [1]

    def reset(self):
        self.t = 0
        return True

    def get_observation(self):
        d = np.zeros((H, W), np.float32)
        d[40:60, 50:70] = 0.05
        cls = (d > 0).astype(np.uint8)
        return {'colormap': np.random.randint(0, 255, (H, W, 3), dtype=np.uint8), 'depthmap': d, 'classmap': cls,
                'idmap': cls.astype(np.float32), 'freespace': np.ones((H, W), np.float32)}

    add_target_pixels = staticmethod(lambda: ([10, 50, 90], [30, 70, 110], None))
    px_to_xy = staticmethod(lambda x, y: [x * 0.005, y * 0.005])
    push = pick = place = lambda self, *a, **k: True
    check_sim = lambda self: True
    check_sorted = lambda self: 1.0 if self.t >= 3 else 0.0
    check_relabel = lambda self: 1.0 if self.t == 2 else 0.0

    def wait_static(self):
        self.t += 1

    def relabel_image(self, colors, ids, id_list, new_class):
        return [c.copy() for c in colors]


def test_replay():
    buf = PrioritizedReplay(10)
    for i in range(12):
        buf.push({}, (0, 0, 0), {}, -1.0, 'push', False)
    assert len(buf) == 10
    _, idx, w = buf.sample(4)
    buf.update(idx, np.array([5.0, 0.0, 1.0, 2.0]))
    assert buf.priorities[idx[0]] > buf.priorities[idx[1]] or idx[0] == idx[1]
    assert w.max() <= 1.0 + 1e-6


def test_train_and_evaluate():
    rearrange_env.Environment = FakeSim  # forked workers inherit the patch
    out = tempfile.mkdtemp()
    os.chdir(out)
    args = argparse.Namespace(
        num_obj=1, max_obj=3, max_actions=5, random_seed=0, logging_directory='t', num_envs=2, gui=False,
        is_testing=False, load_snapshot=False, snapshot=None, snapshot_target=None, save_every=8,
        save_visualizations=False, updates_per_step=1, physics='legacy', randomize=False, max_iterations=24,
        start_method='fork')
    import config
    orig = config.Config
    # tiny batch so the buffer is large enough to optimise within a few steps
    train.Config = lambda **kw: orig(batch_size=4, **kw)
    train.main(args)
    train.Config = orig
    assert os.path.exists('logs/t/episodes.csv')
    models = os.listdir('logs/t/models')
    assert any(m.startswith('snapshot') for m in models), models

    cfg = Config(max_actions=5)
    trainer = Trainer(cfg, is_testing=True)
    venv = VecEnv([partial(rearrange_env.make_env, cfg, False, 1, None, r) for r in range(2)], start_method='fork')
    try:
        res = evaluate.evaluate(trainer, venv, cfg, 1, 5, 100)
    finally:
        venv.close()
    assert res['episodes'] == 5, res
    print('ok:', res)


if __name__ == '__main__':
    test_replay()
    test_train_and_evaluate()
