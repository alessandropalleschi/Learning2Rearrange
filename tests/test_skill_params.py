"""Parameterised skills with a flow-matching parameter policy."""
import math

import numpy as np
import torch

from l2r.config import Config
from l2r.learning.memory import Transition
from l2r.learning.object_centric import Instance
from l2r.learning.skill_params import (action_params, adapt_demo_actions, heading_of, heading_param, push_instance_of,
                                       push_start)

from test_object_centric import observation


def block(r0=40, r1=50, c0=30, c1=50):
    rows, cols = np.mgrid[r0:r1, c0:c1]
    return Instance(0, rows.ravel(), cols.ravel())


def test_push_start_is_behind_the_object_for_the_heading_and_parameters_round_trip():
    inst = block()
    assert push_start(inst, math.pi / 2, (120, 100)) == (40 - 3, 40)  # towards +y (rows up): start before the first row
    assert push_start(inst, 3 * math.pi / 2, (120, 100))[0] == 49 + 3
    assert push_start(inst, math.pi, (120, 100)) in ((44, 49 + 3), (45, 49 + 3))  # towards the back: start in front (centre row 44.5)
    for heading in (math.pi / 2, 0.75 * math.pi, math.pi, 1.25 * math.pi, 1.5 * math.pi):
        assert math.isclose(heading_of(heading_param(heading)), heading, abs_tol=1e-6)
    x = action_params('push', (0, 1, 2, 0.75, math.pi))
    assert np.allclose(x, [0.0, 0.5])
    assert np.allclose(action_params('grasp', (0, 1, 2, math.radians(22.5))), [0.5, 0.0])


def test_a_push_is_matched_to_its_object_and_demos_get_parameters():
    from l2r.learning.object_centric import build_state
    state = build_state(observation(), [16])
    blue = state['instances'][0]
    start = push_start(blue, math.pi, (120, 100))
    assert push_instance_of(state, (0, start[0], start[1], 0.5, math.pi)) == 0
    demos = adapt_demo_actions([Transition({}, (1, 10, 20, 0.4), {}, 0.0, 'push', False),
                                Transition({}, (0, 5, 6), {}, 0.0, 'grasp', False)])
    assert demos[0].action == (0, 10, 20, 0.4, math.pi / 2) and demos[1].action == (0, 5, 6, 0.0)


def test_the_agent_learns_which_heading_and_approach_are_good():
    from l2r.learning.factory import make_trainer
    from l2r.learning.object_centric import crop
    torch.manual_seed(0)
    np.random.seed(0)
    cfg = Config(agent='object', object_algo='iql', skill_params=True, placement='heuristic', hierarchy=True, twin_q=True,
                 batch_size=8, learning_rate=3e-3, prioritized=False, reward_mode='sparse')
    trainer = make_trainer(cfg)
    obs = observation()
    state = trainer.info(obs)
    blue = state['instances'][0]
    _, row, col = state['actions'][0, 0]
    shape = crop(obs['depthmap']).shape
    batch = []  # smooth rewards over the whole parameter ranges: best push straight to the back, best approach +22.5 degrees
    for heading in np.linspace(math.pi / 2, 1.5 * math.pi, 7):
        start = push_start(blue, heading, shape)
        batch.append(Transition(obs, (0, start[0], start[1], 0.5, float(heading)), obs, -5.0 * abs(heading - math.pi) / (math.pi / 2), 'push', True))
    for approach in np.radians(np.linspace(-40, 40, 9)):
        batch.append(Transition(obs, (0, row, col, float(approach)), obs, -1.0 - 5.0 * abs(approach - math.radians(22.5)) / math.radians(45),
                                'grasp', True))
    back = push_start(blue, math.pi, shape)
    for step in range(600):
        trainer.train_step(batch)
        if step % 20 == 19:
            trainer.model_target.load_state_dict(trainer.model.state_dict())
    q = trainer.forward_batch([obs])
    pushes = {k: v for k, v in trainer._chosen.items() if k[0] == 0 and k[1] == 'push'}
    grasps = {k: v for k, v in trainer._chosen.items() if k[0] == 0 and k[1] == 'grasp' and (k[2], k[3]) == (row, col)}
    blue_push = [v for k, v in pushes.items() if push_instance_of(state, (0, k[2], k[3]) + tuple(v)) == 0]
    assert blue_push and abs(blue_push[0][1] - math.pi) < math.radians(35)  # pushes the blue object towards the back
    assert grasps and abs(list(grasps.values())[0][0] - math.radians(22.5)) < math.radians(15)  # the better approach angle
    assert q.push[0, 0].max() > -1000 and q.pick[0, 0][row, col] > -1000
    primitive, pix = 'push', (0, back[0], back[1])
    full = trainer.action_params(0, primitive, pix)
    assert len(full) == 5
