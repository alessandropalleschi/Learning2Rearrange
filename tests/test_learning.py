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
    h = net.initial_hidden(2, (160, 160)) if belief == 'recurrent' else None
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


def test_qnet_without_place_head_skips_it():
    net = QNet(4, 'none', 8, 1, learned_placement=False)
    assert not hasattr(net, 'place_head') and not hasattr(net, 'held_embed')
    x = torch.randn(2, 4, 128, 104)  # the native map size: no padding to a square canvas is needed
    push, pick, place, _ = net(x)
    assert place is None and push.shape == (2, 2, 128, 104) and pick.shape == (2, 1, 128, 104)
    out = net.forward_selected(x, torch.tensor([QNet.PUSH, QNet.PICK]), torch.tensor([1, 0]))
    assert torch.allclose(out[0], push[0, 1:2], atol=1e-4) and torch.allclose(out[1], pick[1], atol=1e-4)


def test_hybrid_backbone_starts_as_the_conv_network_and_is_consistent():
    from l2r.learning.models import TransformerBottleneck, sincos_positions
    codes = sincos_positions(16, 13, 192)
    assert codes.shape == (16 * 13, 192) and len({tuple(c.tolist()) for c in codes}) == 16 * 13  # every cell has its own code
    block = TransformerBottleneck(512, 192, 2)
    x = torch.randn(2, 512, 16, 13)
    assert torch.allclose(block(x), x)  # zero-initialised output projection: identity at the start
    net = QNet(4, 'none', 8, 1, False, 'hybrid', 64, 1)
    for p in net.context.proj_out.parameters():  # make the transformer path active
        torch.nn.init.normal_(p, std=0.05)
    x = torch.randn(2, 4, 128, 104)
    push, pick, _, _ = net(x)
    out = net.forward_selected(x, torch.tensor([QNet.PUSH, QNet.PICK]), torch.tensor([1, 0]))
    assert torch.allclose(out[0], push[0, 1:2], atol=1e-4) and torch.allclose(out[1], pick[1], atol=1e-4)
    (push.mean() + pick.mean()).backward()
    assert net.context.encoder.layers[0].self_attn.in_proj_weight.grad is not None  # gradients reach the attention


def test_pick_head_has_one_channel_per_jaw_yaw_and_trains_the_taken_channel():
    net = QNet(4, 'none', 8, 1, False, n_pick=3)
    x = torch.randn(2, 4, 32, 32)
    assert net(x)[1].shape == (2, 3, 32, 32)
    prim = torch.tensor([QNet.PICK, QNet.PICK])
    out = [net.forward_selected(x, prim, torch.tensor([r, 0])) for r in (0, 2)]
    full = net(x)[1]
    assert torch.allclose(out[0][0, 0], full[0, 0], atol=1e-5) and torch.allclose(out[1][0, 0], full[0, 2], atol=1e-5)


def test_skip_network_is_consistent_and_uses_the_encoder_features():
    torch.manual_seed(0)
    net = QNet(4, 'none', 8, 1, False, skips=True)
    x = torch.randn(2, 4, 128, 104)
    push, pick, _, _ = net(x)
    out = net.forward_selected(x, torch.tensor([QNet.PUSH, QNet.PICK]), torch.tensor([1, 0]))
    assert torch.allclose(out[0], push[0, 1:2], atol=1e-4) and torch.allclose(out[1], pick[1], atol=1e-4)
    (push.mean() + pick.mean()).backward()
    assert net.fuse[0].weight.grad is not None and net.stem[0].weight.grad.abs().sum() > 0
    plain = QNet(4, 'none', 8, 1, False)
    assert sum(p.numel() for p in net.parameters()) > sum(p.numel() for p in plain.parameters())
    # the skips matter: the decoder output changes when only a skip feature changes
    b, (s0, s1, s2) = net.encode_skips(x[:1])
    assert not torch.allclose(net.decode(b, (s0, s1, s2)), net.decode(b, (s0, s1, s2 * 0)))


def test_geometry_channels_mark_the_regions_and_the_coordinates():
    from l2r.learning.trainer import REGION_CHANNELS
    trainer = Trainer(Config(region_channels=True))
    assert trainer.input_channels == 4 + REGION_CHANNELS
    obs = {'colormap': np.zeros((128, 104, 3), np.uint8), 'depthmap': np.zeros((128, 104), np.float32),
           'regions': np.array([[10, 30], [50, 70], [90, 110]])}  # rows of the un-padded map (the maps carry a 4 px border)
    x = trainer.prepare(obs)
    assert x.shape == (4 + REGION_CHANNELS, 128, 104)
    band = x[4 + 0]
    assert band[14:35, :].min() == 1 and band[:14].sum() == 0 and band[35:].sum() == 0  # rows 10..30 + 4 px border
    assert x[4 + 1][54:75].min() == 1 and x[4 + 2][94:115].min() == 1
    rows, cols = x[4 + 3], x[4 + 4]
    assert rows[0, 0] == -1 and rows[-1, 0] == 1 and cols[0, 0] == -1 and cols[0, -1] == 1 and rows[5, 3] == rows[5, 90]
    try:
        trainer.prepare({'colormap': obs['colormap'], 'depthmap': obs['depthmap']})
        raise AssertionError('missing regions must fail loudly')
    except KeyError:
        pass


import pytest  # noqa: E402


@pytest.mark.parametrize('decoder,backbone', [('plain', 'ppm'), ('fpn', 'conv'), ('attn_unet', 'conv'), ('unet_attn4', 'conv'),
                                              ('fpn', 'hybrid'), ('attn_unet', 'ppm'), ('unet', 'aspp'), ('unet', 'ppm_aspp'), ('unet', 'os4+ppm'),
                                              ('unet_attn4', 'os4+aspp')])
def test_decoder_variants_are_consistent_and_trainable(decoder, backbone):
    torch.manual_seed(0)
    stride4 = backbone.startswith('os4+')
    backbone = backbone.replace('os4+', '')
    net = QNet(4, 'none', 8, 1, False, backbone=backbone, transformer_dim=32, transformer_layers=1, decoder=decoder, stride4=stride4)
    if stride4:
        assert net.encode(torch.randn(1, 4, 128, 104)).shape[-2:] == (32, 26)
    for p in net.context.parameters():  # zero-initialised residual outputs (pyramid pooling / transformer): make them active
        if p.dim() > 1 and p.abs().sum() == 0:
            torch.nn.init.normal_(p, std=0.05)
    x = torch.randn(2, 4, 128, 104)
    push, pick, _, _ = net(x)
    assert push.shape == (2, 2, 128, 104) and pick.shape == (2, 1, 128, 104)
    out = net.forward_selected(x, torch.tensor([QNet.PUSH, QNet.PICK]), torch.tensor([1, 0]))
    assert torch.allclose(out[0], push[0, 1:2], atol=1e-4) and torch.allclose(out[1], pick[1], atol=1e-4)
    (push.mean() + pick.mean()).backward()
    names = {'fpn': 'lateral', 'attn_unet': 'gates', 'unet_attn4': 'attn4'}
    if decoder in names:
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in net.named_parameters() if n.startswith(names[decoder]))
    assert net.skips == (decoder != 'plain')


def test_hrnet_keeps_a_full_resolution_branch_and_convnext_encoder_is_consistent():
    torch.manual_seed(0)
    x = torch.randn(2, 4, 128, 104)
    for kwargs in (dict(decoder='hrnet'), dict(decoder='unet', encoder='convnext')):
        net = QNet(4, 'none', 8, 1, False, **kwargs)
        push, pick, _, _ = net(x)
        out = net.forward_selected(x, torch.tensor([QNet.PUSH, QNet.PICK]), torch.tensor([1, 0]))
        assert push.shape == (2, 2, 128, 104)
        assert torch.allclose(out[0], push[0, 1:2], atol=1e-4) and torch.allclose(out[1], pick[1], atol=1e-4)
        (push.mean() + pick.mean()).backward()
        assert all(p.grad is not None for p in net.parameters() if p.requires_grad), kwargs
    hr = QNet(4, 'none', 8, 1, False, decoder='hrnet')
    branches = hr.trunk(x)
    assert [tuple(b.shape[-2:]) for b in branches] == [(128, 104), (64, 52), (32, 26), (16, 13)]  # the full-resolution branch survives


def test_resnet18_encoder_adapts_inputs_freezes_batch_norm_and_is_consistent():
    torch.manual_seed(0)
    net = QNet(9, 'none', 8, 1, False, decoder='unet', encoder='resnet18_scratch')  # 9 input channels: RGB, height, 5 more
    net.train()
    assert all(not m.training for m in net.trunk.modules() if isinstance(m, torch.nn.BatchNorm2d))  # frozen statistics
    x = torch.randn(2, 9, 128, 104)
    push, pick, _, _ = net(x)
    out = net.forward_selected(x, torch.tensor([QNet.PUSH, QNet.PICK]), torch.tensor([1, 0]))
    assert push.shape == (2, 2, 128, 104) and torch.allclose(out[0], push[0, 1:2], atol=1e-4)
    first = net.trunk.pre[0].weight
    assert first.shape == (64, 9, 7, 7) and torch.allclose(first[:, 5], first[:, 3])  # extra channels start identical (mean RGB filter)
    try:
        QNet(4, 'none', 8, 1, False, decoder='plain', encoder='resnet18_scratch')
        raise AssertionError('needs a U-Net style decoder')
    except AssertionError as error:
        assert 'U-Net' in str(error)
