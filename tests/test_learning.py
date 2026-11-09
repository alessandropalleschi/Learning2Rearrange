"""Networks, replay buffers, belief tracker and trainer (no simulator)."""
import numpy as np
import pytest
import torch

from fakes import H, W, FakeSim
from l2r.config import Config
from l2r.learning.belief import BELIEF_CHANNELS, BeliefTracker, compact_belief
from l2r.learning.memory import PrioritizedReplay
from l2r.learning.models import QNet
from l2r.learning.trainer import Trainer


def test_prioritized_replay():
    buffer = PrioritizedReplay(10)
    for _ in range(12):
        buffer.push({}, (0, 0, 0), {}, -1.0, 'push', False)
    assert len(buffer) == 10
    _, _, weights = buffer.sample(4)
    assert weights.max() <= 1.0 + 1e-6
    buffer.update(np.array([0, 1, 2, 3]), np.array([5.0, 0.0, 1.0, 2.0]))  # distinct indices: deterministic
    assert buffer.priorities[0] > buffer.priorities[3] > buffer.priorities[2] > buffer.priorities[1]


def test_belief_tracker_remembers_occluded_cells():
    tracker, sim = BeliefTracker(age_horizon=5), FakeSim()
    first = sim.observe()
    tracker.update(first)
    second = sim.observe()
    second['colormap'][:, 60:] = 105  # cells seen before become occluded
    belief = tracker.update(second, {'primitive': 'push', 'pix': (0, 50, 60), 'place_pix': None})
    assert (belief['rgb'][:, 60:] == first['colormap'][:, 60:]).all()  # remembered
    assert (belief['age'][:, 60:] == 1).all() and (belief['age'][:, 40:50] == 0).all()
    assert belief['push'].sum() > 0
    compact = compact_belief(belief, 5)
    assert compact.shape == (BELIEF_CHANNELS, H, W) and compact.dtype == np.uint8


@pytest.mark.parametrize('belief,channels', [('none', 4), ('spatial', 11), ('recurrent', 4)])
def test_qnet_training_path_matches_inference_path(belief, channels):
    """forward_selected (one pass, used for training) returns the Q-map that forward (all maps) computes."""
    net = QNet(channels, belief, 8)
    x = torch.randn(2, channels, 160, 160)
    h = net.initial_hidden(2, 160) if belief == 'recurrent' else None
    a = torch.zeros(2, 4, 20, 20) if belief == 'recurrent' else None
    held = torch.tensor([[0., 0., 0.], [0., 1., 0.]])
    push, pick, place, h_new = net(x, h, a, held)
    assert push.shape == (2, 2, 160, 160) and pick.shape == place.shape == (2, 1, 160, 160)
    out = net.forward_selected(x, torch.tensor([QNet.PUSH, QNet.PLACE]), torch.tensor([1, 0]), h, a, held)
    assert torch.allclose(out[0], push[0, 1:2], atol=1e-4) and torch.allclose(out[1], place[1], atol=1e-4)
    out = net.forward_selected(x, torch.tensor([QNet.PICK, QNet.PICK]), torch.tensor([0, 0]), h, a, held)
    assert torch.allclose(out, pick, atol=1e-4)
    if belief == 'recurrent':
        assert h_new.shape == h.shape


def test_push_lengths_action_space():
    """Learned push: one Q-map per (direction, length); every channel is what forward_selected returns."""
    from l2r.envs.rearrange import decode_push
    lengths = 3
    net = QNet(4, 'none', 8, lengths)
    x = torch.randn(2, 4, 160, 160)
    push, _, _, _ = net(x)
    assert push.shape == (2, 2 * lengths, 160, 160)
    for index in range(2 * lengths):
        out = net.forward_selected(x, torch.tensor([QNet.PUSH, QNet.PUSH]), torch.tensor([index, index]))
        assert torch.allclose(out[:, 0], push[:, index], atol=1e-4), index
    angle, length = decode_push(4, (0.04, 0.08, 0.12))  # direction 1, second length
    assert abs(angle - np.pi / 2) < 1e-9 and length == 0.08


def test_feasibility_masks_grow_with_push_length():
    trainer = Trainer(Config(push_lengths=(0.04, 0.12)))
    depth = np.zeros((H, W), np.float32)
    depth[40:60, 50:70] = 0.05
    mask = trainer.valid_mask(depth, 'push')
    assert mask.shape == (4, H - 8, W - 4)
    assert mask[1].sum() > mask[0].sum() and (mask[0] == mask[2]).all()  # longer push reaches farther
    maps = trainer.preprocessQmaps(np.zeros((4, H - 8, W - 4), np.float32), depth, 'push')
    assert maps.shape == (4, H - 8, W - 4)
