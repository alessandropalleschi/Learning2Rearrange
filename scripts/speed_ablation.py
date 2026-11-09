"""Physics steps per action, executed fraction and force stops of the simulator speed options (one variant per process:

    PYTHONPATH=. python scripts/speed_ablation.py 60 adaptive+tol3e-3
"""
import sys, time, json, numpy as np, pybullet
from types import SimpleNamespace
steps = [0]; orig = pybullet.stepSimulation
def counting(*a, **k): steps[0] += 1; return orig(*a, **k)
pybullet.stepSimulation = counting
from l2r.config import Config, SimParams
from l2r.envs.rearrange import RearrangeEnv
from l2r.learning.policy import select_for
from l2r.learning.trainer import Trainer
from l2r.sim.workspace import Workspace
base = dict(physics='legacy', randomize=False, step_distance=0.002, move_budget=1000, sim_quality='legacy', camera_scale=1.0,
            force_limit=300.0, force_sense_x=None, adaptive_step=False, velocity_scale=1.0, joint_tolerance=1e-3)
variants = {'legacy': {}, 'adaptive': dict(adaptive_step=True), 'step4mm': dict(step_distance=0.004),
            'tol3e-3': dict(joint_tolerance=3e-3), 'adaptive+tol3e-3': dict(adaptive_step=True, joint_tolerance=3e-3),
            'adaptive+step4mm+tol3e-3': dict(adaptive_step=True, step_distance=0.004, joint_tolerance=3e-3)}
cfg = Config(max_actions=10**6, placement='learned', footprint_placement=True, reward_mode='sparse')
n_actions = int(sys.argv[1]); out = {}
for name, over in [(sys.argv[2], variants[sys.argv[2]])]:
    params = SimParams.from_args(SimpleNamespace(**{**base, **over})); ws = Workspace.from_params(params)
    np.random.seed(5); env = RearrangeEnv(cfg, num_obj=4, params=params); trainer = Trainer(cfg, is_testing=True)
    obs = env.reset(seed=5); steps[0] = 0; n = ex = att = stops = sorted_ = 0; t0 = time.time()
    while n < n_actions:
        q = trainer.forward_batch([obs]); prim, pix, _ = select_for(trainer, obs, q, 0, 1.0, cfg, ws)
        obs, r, te, tr, info = env.step({'primitive': prim, 'pix': pix})
        if info.get('phase') == 'place': continue
        n += 1; ex += int(info['executed']); att += int(info['attempted']); stops += info.get('force_stops', 0)
        if te or tr: sorted_ += int(info['sorted']); obs = env.reset()
    out[name] = dict(steps_per_action=steps[0] / n, executed=ex / max(att, 1), force_stops=stops / n, wall=(time.time() - t0) / n)
    print(name, {k: round(v, 2) for k, v in out[name].items()}, flush=True)
