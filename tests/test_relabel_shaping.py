"""Shaped reward of hindsight-relabelled goal transitions (opt-in)."""
from types import SimpleNamespace

import numpy as np

from l2r.config import Config
from l2r.envs.rearrange import RearrangeEnv
from l2r.sim.constants import LABELS
from l2r.sim.workspace import Workspace


def env_with(**cfg):
    ids = [1, 2, 3, 4, 5]
    scene = SimpleNamespace(object_ids=ids, relabel_ids=ids, relabel_classes=[LABELS['blue']] * 5,
                            class_name=lambda i: 'green',
                            goal_distance=lambda pos, name: 0.0 if pos[0] < 0.5 else 0.1)  # objects 1..3 inside, 4, 5 not
    env = SimpleNamespace(cfg=Config(**cfg), sim=SimpleNamespace(scene=scene), ws=Workspace(),
                          _positions=np.array([[0.0, 0, 0]] * 3 + [[1.0, 0, 0]] * 2))
    env._potential_from = lambda f, d: RearrangeEnv._potential_from(env, f, d)
    return env


def test_relabelled_goal_gets_the_reward_of_a_real_final_step():
    on = env_with(reward_mode='progress', reward_shaping_scale=3.0, relabel_shaping=True)
    assert np.isclose(RearrangeEnv._relabel_reward(on), 0.0 - 3.0 * (3 / 5 - 1))  # 1.2
    assert RearrangeEnv._relabel_reward(env_with(reward_mode='progress', reward_shaping_scale=3.0)) is None  # flag off: reward_goal
    assert RearrangeEnv._relabel_reward(env_with(reward_mode='sparse', relabel_shaping=True)) is None
    assert RearrangeEnv._relabel_reward(env_with(reward_mode='noop', relabel_shaping=True)) is None
