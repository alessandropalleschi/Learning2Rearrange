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

    force_stops = 0
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
    assert w.max() <= 1.0 + 1e-6
    buf.update(np.array([0, 1, 2, 3]), np.array([5.0, 0.0, 1.0, 2.0]))  # distinct indices: deterministic
    assert buf.priorities[0] > buf.priorities[3] > buf.priorities[2] > buf.priorities[1]


class FakePhysics:
    """Stand-in for pybullet's joint sensors: returns a scripted sequence of wrenches."""

    def __init__(self, sequence):
        self.sequence, self.k = sequence, 0

    def getJointState(self, body, joint):
        w = self.sequence[min(self.k, len(self.sequence) - 1)]
        self.k += 1
        return (0, 0, w, 0)


def test_force_monitor():
    from force_sensing import ForceMonitor
    # static offset of 40 N (weight of the tool): must not trigger once tared
    base = [40, 0, 0, 0, 0, 0]
    seq = [base] * 4 + [[40, 0, 400, 0, 0, 0]] + [base] * 6 + [[40, 0, 400, 0, 0, 0]] * 4
    mon = ForceMonitor(FakePhysics(seq), 0, [0], window=3, debounce=2)
    mon.tare(samples=3)
    assert np.allclose(mon.bias[0], base)
    assert not mon.update(100.0)              # offset removed
    assert not mon.update(100.0)              # a single-sample spike (400 N) is rejected by the median filter
    for _ in range(6):
        assert not mon.update(100.0)
    hits = [mon.update(100.0) for _ in range(4)]  # sustained contact is reported
    assert hits[-1] and not hits[0]


def test_suction_and_ik_helpers():
    """Suction hold verification and IK argument construction, against a fake physics backend."""
    import env as real_env
    from config import SimParams

    class P:
        JOINT_PRISMATIC, JOINT_REVOLUTE = 1, 0
        obj_pos = (0.5, 0.0, 0.0)

        def getLinkState(self, body, link):
            return ((0.5, 0.0, 0.05),)

        def getBasePositionAndOrientation(self, body):
            return self.obj_pos, (0, 0, 0, 1)

        def getJointInfo(self, body, j):
            return (j, b'j', 1 if j == 0 else 0, 0, 0, 0, 0, 0, -10000.0 if j == 0 else -3.14, 10000.0 if j == 0 else 3.14)

    fake, old = P(), real_env.p
    real_env.p = fake
    try:
        sim = real_env.Environment.__new__(real_env.Environment)
        sim.params, sim.ur5e, sim.ur5e_ee_id = SimParams(), 0, 9
        sim.activated, sim.contact_constraint = True, 1
        sim._remember_grip(7, ((0.0, 0.0, -0.05), (0, 0, 0, 1)))  # grasped 5 cm below the cup
        assert sim.suction_holds()
        fake.obj_pos = (0.5, 0.0, -0.1)          # the object sagged 5 cm: grasp lost
        assert not sim.suction_holds()
        sim.ur5e_joints, sim.ik_rest_joints = [0, 1, 2], np.zeros(3)
        kw = sim._build_ik_kwargs()
        assert kw['lowerLimits'][0] == -0.5 and kw['upperLimits'][0] == 0.5   # base slide limited to the workspace
        assert kw['jointRanges'][1] == 6.28 and len(kw['restPoses']) == 3
    finally:
        real_env.p = old


def test_push_lengths():
    """Learned push: direction x length action space (network, masks, decoding)."""
    import torch
    from models import QNet
    from rearrange_env import decode_push
    L = 3
    net = QNet(4, 'none', 8, L)
    x = torch.randn(2, 4, 160, 160)
    push, pick, place, _ = net(x)
    assert push.shape == (2, 2 * L, 160, 160)
    for idx in range(2 * L):  # every (direction, length) channel is what forward_selected returns
        out = net.forward_selected(x, torch.tensor([QNet.PUSH, QNet.PUSH]), torch.tensor([idx, idx]))
        assert torch.allclose(out[:, 0], push[:, idx], atol=1e-4), idx
    angle, length = decode_push(4, (0.04, 0.08, 0.12))  # direction 1, second length
    assert abs(angle - np.pi / 2) < 1e-9 and length == 0.08
    cfg = Config(push_lengths=(0.04, 0.12))
    trainer = Trainer(cfg)
    depth = np.zeros((H, W), np.float32)
    depth[40:60, 50:70] = 0.05
    mask = trainer.valid_mask(depth, 'push')
    assert mask.shape == (4, H - 8, W - 4)
    assert mask[1].sum() > mask[0].sum() and (mask[0] == mask[2]).all()  # longer push reaches farther
    assert trainer.preprocessQmaps(np.zeros((4, H - 8, W - 4), np.float32), depth, 'push').shape == (4, H - 8, W - 4)


def test_belief_tracker():
    from belief import BeliefTracker, compact_belief, BELIEF_CHANNELS
    tr = BeliefTracker(age_horizon=5)
    sim = FakeSim()
    o1 = sim.get_observation()
    tr.update(o1)
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


def test_train_and_evaluate(belief='none', placement='heuristic', push_lengths=(0.08,)):
    rearrange_env.Environment = FakeSim  # forked workers inherit the patch
    out = tempfile.mkdtemp()
    os.chdir(out)
    args = argparse.Namespace(
        num_obj=1, max_obj=3, max_actions=5, random_seed=0, logging_directory='t', num_envs=2, gui=False,
        is_testing=False, load_snapshot=False, snapshot=None, snapshot_target=None, save_every=8,
        save_visualizations=False, updates_per_step=1, physics='legacy', randomize=False, max_iterations=24,
        start_method='fork', belief=belief, placement=placement, eps_decay=0.9995, push_lengths=push_lengths,
        step_distance=0.002, move_budget=1000)
    import config
    orig = config.Config
    # tiny batch so the buffer is large enough to optimise within a few steps
    train.Config = lambda **kw: orig(batch_size=4, **kw)
    train.main(args)
    train.Config = orig
    assert os.path.exists('logs/t/episodes.csv')
    models = os.listdir('logs/t/models')
    assert any(m.startswith('snapshot') for m in models), models

    cfg = Config(max_actions=5, belief=belief, placement=placement, push_lengths=tuple(push_lengths))
    trainer = Trainer(cfg, is_testing=True)
    venv = VecEnv([partial(rearrange_env.make_env, cfg, False, 1, None, r) for r in range(2)], start_method='fork')
    try:
        res = evaluate.evaluate(trainer, venv, cfg, 1, 5, 100)
    finally:
        venv.close()
    assert res['episodes'] == 5, res
    print('ok (belief=%s, placement=%s, push_lengths=%s):' % (belief, placement, push_lengths), res)


if __name__ == '__main__':
    test_replay()
    test_force_monitor()
    test_suction_and_ik_helpers()
    test_belief_tracker()
    test_models()
    for mode in ('none', 'spatial', 'recurrent'):
        test_train_and_evaluate(mode)
    test_push_lengths()
    test_learned_placement_env()
    test_train_and_evaluate('none', 'heuristic', (0.04, 0.08, 0.12))
    test_train_and_evaluate('recurrent', 'learned', (0.04, 0.12))
    for mode in ('none', 'spatial', 'recurrent'):
        test_train_and_evaluate(mode, 'learned')
