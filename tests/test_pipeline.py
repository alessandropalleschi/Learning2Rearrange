"""The learning stack end to end with a fake simulator: environment wrapper, parallel workers, training, evaluation."""
import argparse
from functools import partial

import numpy as np
import pytest

import l2r.envs.rearrange as rearrange
from fakes import H, W, FakeSim
from l2r.config import Config
from l2r.envs.vec import VecEnv
from l2r.evaluation import evaluate
from l2r.learning import policy
from l2r.learning.trainer import Trainer
from l2r.sim.workspace import Workspace
from l2r.training import add_arguments, run


@pytest.fixture
def fake_simulator(monkeypatch):
    """Replace the simulator by the fake one (forked workers inherit the patch)."""
    monkeypatch.setattr(rearrange, 'Environment', FakeSim)


def test_learned_placement_is_a_two_step_action(fake_simulator):
    cfg = Config(placement='learned', max_actions=5)
    env = rearrange.RearrangeEnv(cfg)
    assert 'held_class' not in env.reset()
    mid, reward, terminated, _, info = env.step({'primitive': 'grasp', 'pix': (0, 50, 60)})
    assert reward == 0.0 and not terminated and info['phase'] == 'place' and env.actions == 0
    assert mid['held_class'] == 1 and mid['freespace'].shape == (H - 8, W - 4) and len(mid['heuristic_place']) == 2
    with pytest.raises(ValueError):  # while an object is held the next action must be a place
        env.step({'primitive': 'push', 'pix': (0, 50, 60)})
    nxt, reward, _, _, info = env.step({'primitive': 'place', 'pix': (0, 30, 40)})
    assert reward == cfg.reward_step and not env.holding and env.actions == 1 and 'held_class' not in nxt
    # the policy places on free cells only, greedy and exploring
    trainer, ws = Trainer(cfg), Workspace()
    q = trainer.forward_batch([mid])
    for explore in (0.0, 1.0):
        primitive, pix, _ = policy.select_action(trainer, mid, q.push[0], q.pick[0], q.place[0], explore, cfg, ws)
        assert primitive == 'place' and trainer.place_valid(mid)[pix[1], pix[2]]


@pytest.mark.parametrize('belief,placement,push_lengths', [
    ('none', 'heuristic', (0.08,)),
    ('none', 'heuristic', (0.04, 0.08, 0.12)),
    ('spatial', 'learned', (0.04, 0.08)),
    ('recurrent', 'learned', (0.04, 0.12)),
])
def test_train_then_evaluate(fake_simulator, tmp_path, monkeypatch, belief, placement, push_lengths):
    monkeypatch.chdir(tmp_path)
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args(['--num_envs', '2', '--max_obj', '3', '--max_actions', '5', '--save_every', '8',
                              '--max_iterations', '24', '--start_method', 'fork', '--belief', belief,
                              '--placement', placement, '--push_lengths', *map(str, push_lengths),
                              '--logging_directory', 't'])
    monkeypatch.setattr('l2r.training.Config', lambda **kw: Config(batch_size=4, **kw))  # optimise after 4 transitions
    run(args)
    assert (tmp_path / 'logs/t/episodes.csv').exists()
    assert any(p.name.startswith('snapshot') for p in (tmp_path / 'logs/t/models').iterdir())

    cfg = Config(max_actions=5, belief=belief, placement=placement, push_lengths=push_lengths)
    venv = VecEnv([partial(rearrange.make_env, cfg, False, 1, None, r) for r in range(2)], start_method='fork')
    try:
        result = evaluate(Trainer(cfg, is_testing=True), venv, cfg, Workspace(), 1, 5, 100)
    finally:
        venv.close()
    assert result['episodes'] == 5


def test_footprint_placement_keeps_the_object_clear_of_the_back_wall(fake_simulator):
    from l2r.envs.rearrange import clear_of_back_wall, object_depth_px
    idmap = np.zeros((20, 30), np.float32)
    idmap[5:9, 10:18] = 7  # an object 8 columns deep
    assert object_depth_px({'idmap': idmap}, 7) == 8 and object_depth_px({'idmap': idmap}, 3) == 0
    free = clear_of_back_wall(np.ones((20, 30), np.float32), 8, margin=4)
    assert (free[:, :12] == 0).all() and (free[:, 12:] == 1).all()
    env = rearrange.RearrangeEnv(Config(placement='learned', footprint_placement=True))
    env.reset()
    mid = env.step({'primitive': 'grasp', 'pix': (0, 50, 60)})[0]  # the fake object is 20 columns deep
    assert (mid['freespace'][:, :24] == 0).all() and mid['freespace'][:, 24:].any()
