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
        cls = (d > 0).astype(np.uint8)[4:-4, 4:]  # idmap/classmap/freespace are NOT padded (colormap/depthmap are)
        color = np.random.randint(0, 254, (H, W, 3), dtype=np.uint8)
        color[:, :30] = 105  # occluded (grey) cells
        return {'colormap': color, 'depthmap': d, 'classmap': cls,
                'idmap': cls.astype(np.float32), 'freespace': np.ones((H - 8, W - 4), np.float32)}

    add_target_pixels = staticmethod(lambda: ([10, 50, 90], [30, 70, 110], None))
    px_to_xy = staticmethod(lambda x, y: [x * 0.005, y * 0.005])
    push = pick = place = lambda self, *a, **k: True
    check_sim = lambda self: True
    check_sorted = lambda self: 1.0 if self.t >= 3 else 0.0
    check_relabel = lambda self: 1.0 if self.t == 2 else 0.0

    def wait_static(self):
        self.t += 1

    def relabel_image(self, colors, ids, id_list, new_class):
        import env as real_env  # the real implementation, so the padded/un-padded shapes are exercised
        return real_env.Environment.relabel_image(self, colors, ids, id_list, new_class)


def test_replay():
    buf = PrioritizedReplay(10)
    for i in range(12):
        buf.push({}, (0, 0, 0), {}, -1.0, 'push', False)
    assert len(buf) == 10
    _, idx, w = buf.sample(4)
    buf.update(idx, np.array([5.0, 0.0, 1.0, 2.0]))
    assert buf.priorities[idx[0]] > buf.priorities[idx[1]] or idx[0] == idx[1]
    assert w.max() <= 1.0 + 1e-6


def test_belief_tracker():
    from belief import BeliefTracker, compact_belief, BELIEF_CHANNELS
    tr = BeliefTracker(age_horizon=5)
    sim = FakeSim()
    o1 = sim.get_observation()
    b1 = tr.update(o1)
    o2 = sim.get_observation()
    o2['colormap'][:, 60:] = 105  # cells seen before become occluded
    b2 = tr.update(o2, {'primitive': 'push', 'pix': (0, 50, 60), 'place_pix': None})
    assert (b2['rgb'][:, 60:] == o1['colormap'][:, 60:]).all()  # remembered
    assert (b2['age'][:, 60:] == 1).all() and (b2['age'][:, 40:50] == 0).all()
    assert b2['push'].sum() > 0
    c = compact_belief(b2, 5)
    assert c.shape == (BELIEF_CHANNELS, H, W) and c.dtype == np.uint8


def test_models():
    import torch
    from models import QNet
    x = torch.randn(2, 4, 160, 160)
    for belief, ch in (('none', 4), ('spatial', 11), ('recurrent', 4)):
        net = QNet(ch, belief, 8)
        x_ = torch.randn(2, ch, 160, 160)
        h = net.initial_hidden(2, 160) if belief == 'recurrent' else None
        a = torch.zeros(2, 4, 20, 20) if belief == 'recurrent' else None
        held = torch.tensor([[0., 0., 0.], [0., 1., 0.]])
        push, pick, place, h_new = net(x_, h, a, held)
        assert push.shape == (2, 2, 160, 160) and pick.shape == place.shape == (2, 1, 160, 160)
        prim = torch.tensor([QNet.PUSH, QNet.PLACE])
        out = net.forward_selected(x_, prim, torch.tensor([1, 0]), h, a, held)
        assert torch.allclose(out[0], push[0, 1:2], atol=1e-4) and torch.allclose(out[1], place[1], atol=1e-4)
        out = net.forward_selected(x_, torch.tensor([QNet.PICK, QNet.PICK]), torch.tensor([0, 0]), h, a, held)
        assert torch.allclose(out, pick, atol=1e-4)
        if belief == 'recurrent':
            assert h_new.shape == h.shape


def test_learned_placement_env():
    rearrange_env.Environment = FakeSim
    cfg = Config(placement='learned', max_actions=5)
    env = rearrange_env.RearrangeEnv(cfg)
    obs = env.reset()
    assert 'held_class' not in obs
    mid, r, term, trunc, info = env.step({'primitive': 'grasp', 'pix': (0, 50, 60)})
    assert r == 0.0 and not term and info['phase'] == 'place' and env.actions == 0
    assert mid['held_class'] == 1 and mid['freespace'].shape == (H - 8, W - 4) and len(mid['heuristic_place']) == 2
    try:
        env.step({'primitive': 'push', 'pix': (0, 50, 60)})
        raise AssertionError('push while holding must fail')
    except ValueError:
        pass
    nxt, r, term, trunc, info = env.step({'primitive': 'place', 'pix': (0, 30, 40)})
    assert r == cfg.reward_step and not env.holding and env.actions == 1 and 'held_class' not in nxt
    # the policy: only free cells, held object -> place
    import policy
    trainer = Trainer(cfg)
    q = trainer.forward_batch([mid])
    prim, pix, _ = policy.select_action(trainer, mid, q.push[0], q.pick[0], q.place[0], 0.0, cfg)
    assert prim == 'place' and trainer.place_valid(mid)[pix[1], pix[2]]
    prim, pix, _ = policy.select_action(trainer, mid, q.push[0], q.pick[0], q.place[0], 1.0, cfg)
    assert prim == 'place' and trainer.place_valid(mid)[pix[1], pix[2]]
    print('ok: learned placement environment')


def test_train_and_evaluate(belief='none', placement='heuristic'):
    rearrange_env.Environment = FakeSim  # forked workers inherit the patch
    out = tempfile.mkdtemp()
    os.chdir(out)
    args = argparse.Namespace(
        num_obj=1, max_obj=3, max_actions=5, random_seed=0, logging_directory='t', num_envs=2, gui=False,
        is_testing=False, load_snapshot=False, snapshot=None, snapshot_target=None, save_every=8,
        save_visualizations=False, updates_per_step=1, physics='legacy', randomize=False, max_iterations=24,
        start_method='fork', belief=belief, placement=placement, eps_decay=0.9995)
    import config
    orig = config.Config
    # tiny batch so the buffer is large enough to optimise within a few steps
    train.Config = lambda **kw: orig(batch_size=4, **kw)
    train.main(args)
    train.Config = orig
    assert os.path.exists('logs/t/episodes.csv')
    models = os.listdir('logs/t/models')
    assert any(m.startswith('snapshot') for m in models), models

    cfg = Config(max_actions=5, belief=belief, placement=placement)
    trainer = Trainer(cfg, is_testing=True)
    venv = VecEnv([partial(rearrange_env.make_env, cfg, False, 1, None, r) for r in range(2)], start_method='fork')
    try:
        res = evaluate.evaluate(trainer, venv, cfg, 1, 5, 100)
    finally:
        venv.close()
    assert res['episodes'] == 5, res
    print('ok (belief=%s, placement=%s):' % (belief, placement), res)


if __name__ == '__main__':
    test_replay()
    test_belief_tracker()
    test_models()
    for mode in ('none', 'spatial', 'recurrent'):
        test_train_and_evaluate(mode)
    test_learned_placement_env()
    for mode in ('none', 'spatial', 'recurrent'):
        test_train_and_evaluate(mode, 'learned')
