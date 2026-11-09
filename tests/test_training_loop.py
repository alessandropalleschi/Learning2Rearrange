"""Smoke test of the learning stack with a fake simulator (no PyBullet needed).

    python tests/test_training_loop.py
"""
import os
import sys
import types

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
try:
    import pybullet  # noqa: F401
except ImportError:  # stub the simulator modules so rearrange_env can be imported
    for name in ['pybullet', 'pybullet_data', 'pybullet_utils', 'pybullet_utils.bullet_client']:
        sys.modules[name] = types.ModuleType(name)
    sys.modules['pybullet_utils'].bullet_client = sys.modules['pybullet_utils.bullet_client']

import rearrange_env  # noqa: E402
from config import Config  # noqa: E402
from memory import Transition  # noqa: E402
from trainer import Trainer  # noqa: E402
import train  # noqa: E402

H, W = 100, 120


class FakeSim:
    """Minimal stand-in for env.Environment with the same interface RearrangeEnv uses."""

    def __init__(self, gui=False, num_obj=(1,)):
        self.num_obj, self.t, self.id_list, self.new_class = list(num_obj), 0, [1], [1]

    def reset(self):
        self.t = 0
        return True

    def _obs(self):
        d = np.zeros((H, W), np.float32)
        d[40:60, 50:70] = 0.05
        c = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
        cls = (d > 0).astype(np.uint8)
        return {'colormap': c, 'depthmap': d, 'classmap': cls, 'idmap': cls.astype(np.float32),
                'freespace': np.ones((H, W), np.float32)}

    get_observation = _obs
    px_to_xy = staticmethod(lambda x, y: [x * 0.005, y * 0.005])
    xyz_to_pix = staticmethod(lambda p: (int(p[1] / 0.005), int(p[0] / 0.005)))
    add_target_pixels = staticmethod(lambda: ([10, 50, 90], [30, 70, 110], None))
    push = pick = place = lambda self, *a, **k: True
    wait_static = lambda self: True
    check_sim = lambda self: True
    check_sorted = lambda self: 1.0 if self.t >= 3 else 0.0
    check_relabel = lambda self: 1.0 if self.t == 2 else 0.0

    def relabel_image(self, colors, ids, id_list, new_class):
        return [c.copy() for c in colors]


def main():
    rearrange_env.Environment = FakeSim
    cfg = Config(batch_size=4, max_dim=160, max_actions=5)
    env = rearrange_env.RearrangeEnv(cfg, num_obj=1)
    trainer = Trainer(cfg)
    obs = env.reset()
    losses, relabels = [], 0
    for it in range(8):
        env.sim.t += 1
        push_q, pick_q = trainer.forward(obs['colormap'], obs['depthmap'])
        assert push_q.shape == (2, H - 8, W - 4) and pick_q.shape == (1, H - 8, W - 4), push_q.shape
        prim, pix, _ = train.select_action(trainer, env, obs, push_q, pick_q, 1.0, False, cfg)
        nxt, r, term, trunc, info = env.step({'primitive': prim, 'pix': pix})
        assert r in (cfg.reward_step, cfg.reward_goal, cfg.reward_invalid)
        t = Transition(obs, pix, nxt, r, prim, term)
        trainer.buffer.push(*t)
        losses.append(trainer.train_step([t]))
        if info['relabel']:
            relabels += 1
            o2, n2 = env.relabel(obs, nxt)
            trainer.buffer.push(o2, pix, n2, cfg.reward_goal, prim, True)
        trainer.optimize()
        obs = env.reset() if (term or trunc) else nxt
        if term or trunc:
            env.sim.t = 0
    assert all(np.isfinite(losses)), losses
    print('ok: %d steps, losses %s, relabels %d' % (len(losses), np.round(losses, 3), relabels))


if __name__ == '__main__':
    main()
