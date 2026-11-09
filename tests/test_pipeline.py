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


@pytest.mark.parametrize('belief,placement,push_lengths,extra', [
    ('none', 'heuristic', (0.08,), ['--hierarchy']),
    ('none', 'heuristic', (0.08,), ['--backbone', 'hybrid']),
    ('none', 'heuristic', (0.08,), []),
    ('none', 'heuristic', (0.04, 0.08, 0.12), []),
    ('spatial', 'learned', (0.04, 0.08), []),
    ('recurrent', 'learned', (0.04, 0.12), []),
])
def test_train_then_evaluate(fake_simulator, tmp_path, monkeypatch, belief, placement, push_lengths, extra):
    monkeypatch.chdir(tmp_path)
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args(['--num_envs', '2', '--max_obj', '3', '--max_actions', '5', '--save_every', '8',
                              '--max_iterations', '24', '--start_method', 'fork', '--belief', belief,
                              '--placement', placement, '--push_lengths', *map(str, push_lengths),
                              '--logging_directory', 't', '--batch_size', '4', *extra])
    run(args)
    assert (tmp_path / 'logs/t/episodes.csv').exists()
    assert any(p.name.startswith('snapshot') for p in (tmp_path / 'logs/t/models').iterdir())

    cfg = Config.from_args(parser.parse_args(['--max_actions', '5', '--belief', belief, '--placement', placement,
                                              '--push_lengths', *map(str, push_lengths), *extra]))
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


def run_fake_episode(**cfg_kwargs):
    """Three actions to the goal on the fake simulator; returns the rewards, the shaping terms and the config."""
    cfg = Config(max_actions=10, **cfg_kwargs)
    env = rearrange.RearrangeEnv(cfg)
    env.reset()
    rewards, shaping, relabels = [], [], []
    for _ in range(3):
        _, reward, terminated, _, info = env.step({'primitive': 'push', 'pix': (0, 50, 60)})
        rewards.append(reward)
        shaping.append(info['shaping'])
        relabels.append(info['relabel'])
    assert terminated
    return cfg, np.array(rewards), np.array(shaping), relabels


@pytest.mark.parametrize('mode,phi0', [('progress', -1.0), ('distance', -0.2 / 0.6)])
def test_potential_based_shaping_keeps_the_discounted_return_up_to_the_initial_potential(fake_simulator, mode, phi0):
    """sum_t gamma^t (r_t + gamma phi(s_t+1) - phi(s_t)) = sum_t gamma^t r_t - phi(s_0) for an absorbing goal (phi = 0):
    the shaped return differs from the sparse one by a constant, so the optimal policy is the same."""
    cfg, shaped, extra, _ = run_fake_episode(reward_mode=mode)
    _, sparse, no_extra, _ = run_fake_episode(reward_mode='sparse')
    assert np.allclose(sparse, [-1, -1, 0]) and np.allclose(no_extra, 0)
    discount = cfg.gamma ** np.arange(3)
    assert np.isclose((discount * shaped).sum(), (discount * sparse).sum() - phi0)
    assert (extra[:2] > 0).all()  # the fake sorting makes progress with every action: positive shaping


def test_noop_penalty_applies_only_when_nothing_moved(fake_simulator, monkeypatch):
    _, moving, extra, _ = run_fake_episode(reward_mode='noop', noop_penalty=2.0)
    assert np.allclose(extra, 0)
    monkeypatch.setattr(FakeSim, '_positions', lambda self: np.zeros((1, 3)))  # nothing ever moves
    _, still, extra, _ = run_fake_episode(reward_mode='noop', noop_penalty=2.0)
    assert np.allclose(extra, [-2.0, -2.0, 0.0])  # not on the goal-reaching action (terminal)


def test_relabelling_still_triggers_with_shaped_rewards_and_the_value_range_follows_the_reward(fake_simulator):
    _, _, _, relabels = run_fake_episode(reward_mode='progress')
    assert relabels == [False, True, False]  # the fake scene allows relabelling at its 2nd action
    assert Config().q_range == pytest.approx((-100.0, 0.0))
    assert Config(reward_mode='progress', reward_shaping_scale=2.0).q_range == pytest.approx((-100.0, 2.0))
    assert Config(reward_mode='noop', noop_penalty=1.0).q_range[0] == pytest.approx(-110.0)
