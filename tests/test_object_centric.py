"""Object-centric agent: instances from the observation alone, candidate slots, and a bandit the trainer must solve."""
import numpy as np
import torch

from l2r.config import Config
from l2r.learning.memory import Transition
from l2r.learning.object_centric import CLASS_RGB, ObjectQNet, build_state, collate, rasterize, slot_of
from l2r.learning.object_trainer import ObjectTrainer

H, W = 128, 104


def observation(blue_rows=(35, 46), green_rows=(75, 86)):
    """Padded maps (4 px border) with a blue object that is NOT in its region and a green one that is."""
    color = np.full((H, W, 3), 255, np.uint8)
    depth = np.zeros((H, W), np.float32)
    color[4 + blue_rows[0]:4 + blue_rows[1], 4 + 30:4 + 50] = CLASS_RGB[0]
    depth[4 + blue_rows[0]:4 + blue_rows[1], 4 + 30:4 + 50] = 0.05
    color[4 + green_rows[0]:4 + green_rows[1], 4 + 60:4 + 80] = CLASS_RGB[1]
    depth[4 + green_rows[0]:4 + green_rows[1], 4 + 60:4 + 80] = 0.05
    color[4 + 35:4 + 46, 4:4 + 30] = 105  # grey cells behind the blue object
    return {'colormap': color, 'depthmap': depth, 'regions': np.array([[10, 30], [70, 90], [100, 115]])}


def test_instances_features_and_candidates_come_from_the_observation_only():
    state = build_state(observation(), [16])
    assert [(i.cls, i.rows.size) for i in state['instances']] == [(0, 220), (1, 220)]
    tokens = state['tokens']
    assert tokens.shape == (1 + 3 + 2, 18)
    assert tokens[4, 12] == 0 and tokens[5, 12] == 1  # blue: misplaced, green: inside its region
    assert tokens[4, 15] > 0.5 and tokens[5, 15] == 0  # the blue object has grey (hidden) cells behind it
    pick = state['actions'][0, 0]
    assert tuple(pick[1:]) == (40, 47)  # front face is column 49: aimed 2 cells inside, in the middle of the width
    assert state['valid'].shape == (2, 3) and state['valid'].all()
    assert slot_of(state, 'grasp', (0, 41, 48)) == (0, 0) and slot_of(state, 'grasp', (0, 100, 5)) is None
    assert slot_of(state, 'place', (0, 40, 47)) is None  # placement is not a decision of this agent


def test_a_push_slot_needs_an_object_in_the_blades_path():
    obs = observation()
    state = build_state(obs, [4])  # a 4-cell push reaches nothing from 3 cells beside the object... except along its width
    assert state['valid'][:, 0].all()  # the pick is always valid
    far = build_state(obs, [1])
    assert not far['valid'][:, 1:].any()  # a 1-cell push from 3 cells away touches nothing


def test_network_and_rasterisation():
    state = build_state(observation(), [16])
    tokens, n_objects, valid = collate([state, state], 'cpu')
    q = ObjectQNet(n_slots=3)(tokens, n_objects)
    assert q.shape == (2, 2, 3) and valid.shape == (2, 2, 3)
    push, pick = rasterize(state, q[0].detach().numpy(), (120, 100), 2)
    assert push.shape == (2, 120, 100) and pick.shape == (1, 120, 100)
    assert (pick > -999).sum() == 2 and (push > -999).sum() == 4  # only the candidate pixels carry values


def test_trainer_learns_which_object_to_grasp_and_acts_on_it():
    torch.manual_seed(0)
    np.random.seed(0)
    cfg = Config(placement='heuristic', batch_size=8, learning_rate=3e-3, prioritized=False, reward_mode='sparse')
    trainer = ObjectTrainer(cfg)
    obs = observation()
    state = trainer.info(obs)
    good, bad = tuple(state['actions'][0, 0]), tuple(state['actions'][1, 0])  # grasp the misplaced blue / the placed green
    batch = [Transition(obs, good, obs, 0.0, 'grasp', True)] * 4 + [Transition(obs, bad, obs, -5.0, 'grasp', True)] * 4
    for _ in range(150):
        trainer.train_step(batch)
    q = trainer.forward_batch([obs])
    pick = q.pick[0, 0]
    assert pick[good[1], good[2]] > pick[bad[1], bad[2]] + 2  # about 0 against about -5
    assert np.unravel_index(np.argmax(pick), pick.shape) == (good[1], good[2])
    assert trainer.last_td.shape == (8,)
