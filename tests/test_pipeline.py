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
    cfg = Config(placement='learned', max_actions=5, reward_mode='sparse')  # the step reward of the paper
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
    ('none', 'heuristic', (0.08,), ['--agent', 'object']),
    ('spatial', 'learned', (0.08,), ['--agent', 'object', '--hierarchy']),  # learned placement + continuous length + ghost tokens
    ('spatial', 'learned', (0.08,), ['--agent', 'object', '--hierarchy', '--outcome_weight', '0.5', '--outcome_penalty', '0.3']),
    ('spatial', 'learned', (0.08,), ['--agent', 'object', '--hierarchy', '--twin_q', '--n_step', '3']),
    ('spatial', 'learned', (0.08,), ['--agent', 'object', '--hierarchy', '--object_algo', 'iql', '--twin_q', '--n_step', '3']),
    ('spatial', 'learned', (0.08,), ['--agent', 'object', '--hierarchy', '--object_algo', 'iql', '--skill_params', '--twin_q']),
    ('spatial', 'learned', (0.08,), ['--agent', 'object', '--hierarchy', '--object_algo', 'iql', '--history_tokens']),
    ('spatial', 'learned', (0.08,), ['--agent', 'object', '--hierarchy', '--object_algo', 'iql', '--twin_q', '--rudder', '0.5', '--n_step', '3']),
    ('none', 'learned', (0.08,), ['--arch_skips', '--rudder', '0.5']),
    ('spatial', 'learned', (0.08,), ['--agent', 'object', '--hierarchy', '--outcome_weight', '0.5', '--outcome_mask', '0.3']),
    ('spatial', 'learned', (0.08,), ['--agent', 'object', '--hierarchy', '--object_algo', 'iql', '--outcome_weight', '0.5', '--outcome_mask', '0.3']),
    ('none', 'learned', (0.08,), ['--arch_skips', '--n_step', '3']),
    ('none', 'learned', (0.08,), ['--agent', 'policy_critic']),
    ('none', 'learned', (0.08,), ['--arch_skips', '--region_channels', '--equiv_targets']),
    ('none', 'heuristic', (0.08,), ['--end_effector', 'gripper', '--jaw_yaws', '3']),
    ('none', 'learned', (0.08,), ['--end_effector', 'gripper', '--jaw_yaws', '3', '--hierarchy']),
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


def test_footprint_placement_fits_the_whole_object_into_free_space():
    from l2r.envs.rearrange import fit_footprint, object_extent_px
    idmap = np.zeros((20, 30), np.float32)
    idmap[5:13, 10:18] = 7  # 8 cells deep (x), 8 cells wide (y)
    assert object_extent_px({'idmap': idmap}, 7) == (8, 8) and object_extent_px({'idmap': idmap}, 3) == (0, 0)
    free = np.ones((40, 60), np.uint8)
    free[18:22, 30:40] = 0  # an obstacle
    fit = fit_footprint(free, depth_px=10, width_px=8)
    rows, cols = np.nonzero(fit)
    assert rows.min() >= 6 and cols.min() >= 13  # walls: half the width + margin from the sides, depth + margin from the back
    for r, c in zip(rows, cols):  # the whole footprint of every offered cell is free
        assert (free[r - 6:r + 6, c - 13:c + 1] > 0).all()
    assert fit[20, 35] == 0 and fit[20, 45] == 0 and fit[20, 55] == 1  # not on / behind-onto the obstacle, free behind it
    assert (fit_footprint(np.zeros((10, 10), np.uint8), 4, 4) == 0).all()  # nothing fits: the plain map comes back


def test_footprint_placement_is_applied_to_the_held_object(fake_simulator):
    for space in ('depth', 'region'):
        env = rearrange.RearrangeEnv(Config(placement='learned', footprint_placement=True, placement_space=space))
        env.reset()
        free = env.step({'primitive': 'grasp', 'pix': (0, 50, 60)})[0]['freespace']  # the fake object is 20 x 20 cells
        assert free[:, :23].sum() == 0 and free.any()  # both keep the back wall clear
        if space == 'depth':
            assert free[:10].any() and free[-10:].any()  # the original rule does not look at the side margins
        else:
            assert free[:10].sum() == 0 and free[30:].sum() == 0  # only the object's own target region (rows 10..30)



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


@pytest.mark.parametrize('mode,phi0', [('progress', -1.0), ('distance', -0.2 / 0.6), ('combined', -1.0 - 0.2 / 0.6)])
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
    assert Config(reward_mode='sparse').q_range == pytest.approx((-100.0, 0.0))
    assert Config(reward_mode='progress', reward_shaping_scale=2.0).q_range == pytest.approx((-100.0, 2.0))
    assert Config(reward_mode='noop', noop_penalty=1.0).q_range[0] == pytest.approx(-110.0)
    assert Config(reward_mode='combined', reward_shaping_scale=3.0, noop_penalty=0.5).q_range == pytest.approx((-105.0, 6.0))


def test_combined_reward_adds_the_noop_penalty_to_the_potential_shaping(fake_simulator, monkeypatch):
    monkeypatch.setattr(FakeSim, '_positions', lambda self: np.zeros((1, 3)))  # nothing ever moves
    _, _, with_noop, _ = run_fake_episode(reward_mode='combined', noop_penalty=0.5)
    _, _, potential_only, _ = run_fake_episode(reward_mode='combined', noop_penalty=0.0)
    assert np.allclose(with_noop - potential_only, [-0.5, -0.5, 0.0])


def test_curriculum_advances_after_a_greedy_validation(fake_simulator, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args(['--num_envs', '2', '--max_obj', '3', '--max_actions', '5', '--max_iterations', '60',
                              '--start_method', 'fork', '--batch_size', '4', '--logging_directory', 'v',
                              '--curriculum_eval', '3', '--curriculum_eval_every', '4'])
    run(args)
    out = capsys.readouterr().out
    assert 'Validation 1 (1 objects, 3 greedy episodes): success 1.00' in out and 'Curriculum: now using 2 objects' in out
    rows = (tmp_path / 'logs/v/validation.csv').read_text().strip().splitlines()
    assert rows[0].split(',')[2:] == ['1', '1.000']


def test_pooled_validation_waits_for_the_window_and_fixed_seeds_repeat_the_scenes(fake_simulator, tmp_path, monkeypatch, capsys):
    from l2r import evaluation
    monkeypatch.chdir(tmp_path)
    seeds = []
    real = evaluation.evaluate

    def spy(trainer, venv, cfg, ws, num_obj, episodes, base_seed):
        seeds.append((num_obj, base_seed))
        return real(trainer, venv, cfg, ws, num_obj, episodes, base_seed)
    monkeypatch.setattr(evaluation, 'evaluate', spy)
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args(['--num_envs', '2', '--max_obj', '3', '--max_actions', '5', '--max_iterations', '80',
                              '--start_method', 'fork', '--batch_size', '4', '--logging_directory', 'w',
                              '--curriculum_eval', '3', '--curriculum_eval_every', '4', '--curriculum_window', '2',
                              '--fixed_validation_seeds', '--success_threshold', '0.85'])
    run(args)
    out = capsys.readouterr().out
    assert 'Validation 1 (1 objects, 3 greedy episodes): success 1.00, mean actions 3.0, pooled over 1: 1.00 -> stay' in out
    assert 'pooled over 2: 1.00 -> advance' in out  # the fake scene is always sorted: advances once the window is full
    first_stage = [seed for n, seed in seeds if n == 1]
    assert len(first_stage) == 2 and len(set(first_stage)) == 1  # the same scenes at both validations of the stage


def test_stage_mixing_draws_earlier_stages_and_logs_the_actual_object_count(fake_simulator, tmp_path, monkeypatch):
    from l2r.training import TrainingLoop
    monkeypatch.chdir(tmp_path)
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args(['--num_envs', '2', '--start_method', 'fork', '--logging_directory', 'm', '--stage_mix', '0.5'])
    loop = TrainingLoop(args)
    try:
        loop.curriculum.num_obj = 4
        counts = loop._draw_objects(400)
        assert set(counts) == {1, 2, 3, 4} and 0.35 < np.mean([c < 4 for c in counts]) < 0.65
        assert loop._draw_objects(400, mix=False) == [4] * 400  # the first reset uses the current stage only
        loop.curriculum.num_obj = 1
        assert set(loop._draw_objects(50)) == {1}  # nothing earlier to mix in
    finally:
        loop.venv.close()
        loop.logger.close()


def test_environment_and_fake_follow_the_simulator_boundary():
    from l2r.sim.backend import SimBackend
    from l2r.sim.environment import Environment
    methods = [n for n, v in vars(SimBackend).items() if callable(v) and not n.startswith('_')]
    assert {'reset', 'observe', 'push', 'pick', 'place', 'wait_static', 'is_valid'} <= set(methods)
    for cls in (Environment, FakeSim):
        missing = [n for n in methods if not hasattr(cls, n)]
        assert not missing, (cls.__name__, missing)


def test_placement_space_falls_back_when_the_footprint_does_not_fit_in_the_target_region(fake_simulator):
    from l2r.envs.rearrange import placement_space
    sim = FakeSim()  # target rows of class label 1 (blue): 10..30
    free = np.zeros((120, 100), np.uint8)
    free[12:28, 30:90] = 1  # the region has room for a 2 x 2 footprint ...
    free[60:110, 30:90] = 1  # ... and so does the space outside it
    fits = placement_space(sim, 1, free, (4, 4))
    assert fits[10:30].any()  # the footprint fits in the region: the eroded map is used
    tight = np.zeros((120, 100), np.uint8)
    tight[11:14, 30:90] = 1  # a crowded region: too tight for a 20-cell wide object ...
    tight[60:110, 30:90] = 1  # ... but plenty of room elsewhere
    out = placement_space(sim, 1, tight, (10, 20))
    assert out[10:30].any()  # fell back: the target region still offers cells (back-wall rule only)
    assert out[:, :14].sum() == 0  # and the back wall is still kept clear


def test_placement_space_offers_only_cells_inside_the_target_region(fake_simulator):
    from l2r.envs.rearrange import placement_space
    free = np.ones((120, 100), np.uint8)
    out = placement_space(FakeSim(), 1, free, (4, 4))  # class 1: rows 10..30
    assert out[10:30].any() and out[:10].sum() == 0 and out[30:].sum() == 0
    none = np.zeros((120, 100), np.uint8)
    none[60:110, 30:90] = 1  # no free cell in the region at all: everything else is offered (nothing else to do)
    assert placement_space(FakeSim(), 1, none, (4, 4))[60:].any()


def test_observations_record_the_last_action_and_its_outcome(fake_simulator):
    from l2r.config import Config
    from l2r.envs.rearrange import LAST_ACTION_SIZE, RearrangeEnv
    env = RearrangeEnv(Config(placement='heuristic'), num_obj=1)
    obs = env.reset()
    assert obs['last_action'].shape == (LAST_ACTION_SIZE,) and not obs['last_action'].any() and tuple(obs['last_pix']) == (-1, -1)
    obs = env.step({'primitive': 'push', 'pix': (0, 50, 60)})[0]
    assert obs['last_action'][0] == 1 and tuple(obs['last_pix']) == (50, 60)
    assert 0.0 <= obs['last_action'][5] <= 1.0


def test_parameterised_actions_reach_the_simulator(fake_simulator):
    from unittest import mock

    from l2r.config import Config
    from l2r.envs.rearrange import RearrangeEnv
    env = RearrangeEnv(Config(placement='heuristic'), num_obj=1)
    env.reset()
    with mock.patch.object(env.sim, 'push', return_value=True) as push, mock.patch.object(env.sim, 'pull', return_value=True) as pull, \
            mock.patch.object(env.sim, 'pick', return_value=False) as pick:
        env.step({'primitive': 'push', 'pix': (0, 50, 60), 'heading': 3.0, 'length': 0.05})
        assert push.call_args.kwargs == {'push_length': 0.05, 'heading': 3.0}
        env.step({'primitive': 'pull', 'pix': (0, 50, 60), 'heading': 0.2, 'length': 0.07, 'approach': 0.1})
        assert pull.call_args.kwargs == {'approach': 0.1, 'heading': 0.2, 'distance': 0.07}
        env.step({'primitive': 'grasp', 'pix': (0, 50, 60), 'approach': -0.4})
        assert pick.called and pick.call_args.kwargs['approach'] == -0.4


def test_parameters_in_the_action_tuple_reach_the_simulator(fake_simulator):
    from unittest import mock

    from l2r.config import Config
    from l2r.envs.rearrange import RearrangeEnv
    cfg = Config(placement='heuristic')
    env = RearrangeEnv(cfg, num_obj=1)
    env.reset()
    with mock.patch.object(env.sim, 'push', return_value=True) as push, mock.patch.object(env.sim, 'pick', return_value=False) as pick:
        env.step({'primitive': 'push', 'pix': (0, 50.0, 60.0, 0.5, 3.0)})
        low, high = cfg.push_length_range
        assert push.call_args.kwargs == {'push_length': low + 0.5 * (high - low), 'heading': 3.0}
        env.step({'primitive': 'grasp', 'pix': (0, 50, 60, -0.3)})
        assert pick.call_args.kwargs['approach'] == -0.3
