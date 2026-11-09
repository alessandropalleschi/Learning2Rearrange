"""Q-network: one shared fully-convolutional backbone with a push head and a pick head.

Improvements over the original two independent networks:
  * 4-channel input (RGB + height), as described in the paper (was 6 channels, depth repeated 3x).
  * GroupNorm instead of BatchNorm: statistics do not depend on the (tiny, correlated) batch.
  * Global-context block at the bottleneck: pooled features of the whole cabinet are fed back to every
    pixel, so each Q value can depend on the full scene (fully convolutional nets only see a local window).
  * Shared backbone: the push (rotation 0) and pick Q-maps reuse the same features of the un-rotated
    input, so one forward pass serves both primitives.

Belief state (ablation, `belief` argument of QNet):
  * 'none'       memoryless policy on the current observation (the paper).
  * 'spatial'    extra input channels with a persistent memory map (see belief.py): in_channels = 4 + 7.
  * 'recurrent'  ConvGRU at the bottleneck. Its hidden state is the latent belief; it is updated from the
                 encoded latest observation and the encoded latest action, and fed back into the features
                 of both pushing directions. The state is computed in the un-rotated frame; the 180 degree
                 branch receives it flipped.

Pushing directions: the paper models two (left/right) = 0 and 180 degree rotations of the input. A 180
degree rotation of a square image is an exact double flip, so no interpolation is needed.
"""
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IN_CHANNELS = 4  # R, G, B, height


def norm(channels, groups=8):
    return nn.GroupNorm(min(groups, channels), channels)


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, 3, padding=1, bias=False)
        self.norm1 = norm(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, 3, padding=1, bias=False)
        self.norm2 = norm(out_planes)
        self.shortcut = None if in_planes == out_planes else nn.Conv2d(in_planes, out_planes, 1, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        identity = x if self.shortcut is None else self.shortcut(x)
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return F.relu(out + identity)


class GlobalContext(nn.Module):
    """Pools the feature map (avg + max) into a vector and adds a learned projection back to every pixel."""

    def __init__(self, channels):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2 * channels, channels), nn.ReLU(inplace=True),
                                 nn.Linear(channels, channels))

    def forward(self, x):
        g = torch.cat([x.mean(dim=(2, 3)), x.amax(dim=(2, 3))], dim=1)
        return x + self.mlp(g)[:, :, None, None]


def sincos_positions(height, width, dim):
    """Fixed 2D sine-cosine position codes, (height * width, dim): half of the channels encode the row, half the column."""
    assert dim % 4 == 0
    quarter = dim // 4
    freq = 1.0 / (10000 ** (torch.arange(quarter) / quarter))
    rows, cols = torch.arange(height)[:, None] * freq[None], torch.arange(width)[:, None] * freq[None]
    row_code = torch.cat([rows.sin(), rows.cos()], 1)[:, None, :].expand(height, width, 2 * quarter)
    col_code = torch.cat([cols.sin(), cols.cos()], 1)[None, :, :].expand(height, width, 2 * quarter)
    return torch.cat([row_code, col_code], -1).reshape(height * width, dim)


class TransformerBottleneck(nn.Module):
    """Global self-attention over the bottleneck grid (a token per 8x8 pixel block, 16x13 = 208 tokens for the 128x104
    maps): lets every location look at all objects, target regions and occlusions at once, which convolutions only do
    through their receptive field. Pre-norm encoder layers; the output projection starts at zero, so the network starts
    as the convolutional one."""

    def __init__(self, channels=512, dim=192, layers=3, heads=4):
        super().__init__()
        self.dim = dim
        self.proj_in = nn.Conv2d(channels, dim, 1)
        layer = nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout=0.0, activation='gelu', batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.proj_out = nn.Conv2d(dim, channels, 1)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)
        self._positions = {}

    def forward(self, x):
        b, _, h, w = x.shape
        key = (h, w, x.device)
        if key not in self._positions:
            self._positions[key] = sincos_positions(h, w, self.dim).to(x.device)
        tokens = self.proj_in(x).flatten(2).transpose(1, 2) + self._positions[key]
        tokens = self.norm(self.encoder(tokens))
        return x + self.proj_out(tokens.transpose(1, 2).reshape(b, self.dim, h, w))


HELD_CLASSES = {1: 0, 3: 1, 5: 2}  # class label (blue, green, yellow) -> one-hot index of the held object
ACTION_CHANNELS = 4  # last action encoding at bottleneck resolution: push 0deg, push 180deg, pick, place


class ConvGRUCell(nn.Module):
    """Convolutional GRU: h' = (1-z)*h + z*tanh(W[x, r*h])."""

    def __init__(self, in_ch, hid_ch, k=3):
        super().__init__()
        self.hid_ch = hid_ch
        self.gates = nn.Conv2d(in_ch + hid_ch, 2 * hid_ch, k, padding=k // 2)
        self.cand = nn.Conv2d(in_ch + hid_ch, hid_ch, k, padding=k // 2)

    def forward(self, x, h):
        z, r = torch.sigmoid(self.gates(torch.cat([x, h], 1))).chunk(2, dim=1)
        n = torch.tanh(self.cand(torch.cat([x, r * h], 1)))
        return (1 - z) * h + z * n


Outputs = namedtuple('Outputs', ['push', 'pick', 'place', 'hidden', 'lengths', 'prim'])


class QNet(nn.Module):
    nr_rotations = 2  # push left / right (0 and 180 degrees)

    def __init__(self, in_channels=IN_CHANNELS, belief='none', hidden=64, n_push_lengths=1, learned_placement=True,
                 backbone='conv', transformer_dim=192, transformer_layers=3, hierarchical=False, n_pick=1, skips=False):
        """learned_placement: build the place head (conditioned on the held class). Without it `forward` returns
        `None` as place map: the head would cost about as much as the pick head for nothing.
        n_pick: pick Q-map channels (parallel-jaw gripper: one per jaw yaw; suction cup: 1)."""
        super().__init__()
        self.n_pick = n_pick
        self.belief, self.hidden, self.n_push_lengths = belief, hidden, n_push_lengths
        self.learned_placement, self.backbone, self.hierarchical = learned_placement, backbone, hierarchical
        self.skips = skips
        self.stem = nn.Sequential(nn.Conv2d(in_channels, 64, 3, padding=1, bias=False), norm(64), nn.ReLU(inplace=True))
        self.rb1, self.rb2, self.rb3 = ResidualBlock(64, 128), ResidualBlock(128, 256), ResidualBlock(256, 512)
        # bottleneck: pooled global context (conv backbone) or global self-attention (hybrid conv + transformer)
        self.context = (TransformerBottleneck(512, transformer_dim, transformer_layers) if backbone == 'hybrid'
                        else GlobalContext(512))
        if belief == 'recurrent':
            self.rec_in = nn.Conv2d(512 + ACTION_CHANNELS, hidden, 1)
            self.gru = ConvGRUCell(hidden, hidden)
            self.rec_out = nn.Conv2d(hidden, 512, 1)
            nn.init.zeros_(self.rec_out.weight)  # start as a memoryless network, learn to use the belief
            nn.init.zeros_(self.rec_out.bias)
        if skips:  # U-Net style: the decoder also sees the encoder features of the same resolution (1/4, 1/2, full)
            self.rb4, self.rb5, self.rb6 = ResidualBlock(512, 256), ResidualBlock(256 + 256, 128), ResidualBlock(128 + 128, 64)
            self.fuse = nn.Sequential(nn.Conv2d(64 + 64, 64, 3, padding=1, bias=False), norm(64), nn.ReLU(inplace=True))
        else:
            self.rb4, self.rb5, self.rb6 = ResidualBlock(512, 256), ResidualBlock(256, 128), ResidualBlock(128, 64)
        # one Q-map per selectable push length; the second direction is the same head on the flipped input
        if hierarchical:
            # hierarchical parameterised action: a *continuous* push length. The push critic gets the length as an extra
            # input channel, an actor head proposes it per pixel, a primitive head scores push / pick / place globally.
            self.push_head = nn.Sequential(ResidualBlock(64 + 1, 64), nn.Conv2d(64, 1, 1))
            self.length_head = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 1, 1))
            self.prim_head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Linear(64, 3))
        else:
            self.push_head = nn.Sequential(ResidualBlock(64, 64), nn.Conv2d(64, n_push_lengths, 1))
        self.pick_head = nn.Sequential(ResidualBlock(64, 64), nn.Conv2d(64, n_pick, 1))
        if learned_placement:  # Q-map over the free cells, conditioned on the class of the object in the gripper
            self.place_head = nn.Sequential(ResidualBlock(64, 64), nn.Conv2d(64, 1, 1))
            self.held_embed = nn.Linear(len(HELD_CLASSES), 512)
            nn.init.zeros_(self.held_embed.weight)
            nn.init.zeros_(self.held_embed.bias)

    @staticmethod
    def _up(x):
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

    def encode_skips(self, x):
        """Input -> (bottleneck features (B,512,H/8,W/8), encoder features at full, 1/2 and 1/4 resolution for the skips)."""
        s0 = self.stem(x)
        s1 = self.rb1(F.max_pool2d(s0, 2))
        s2 = self.rb2(F.max_pool2d(s1, 2))
        return self.rb3(F.max_pool2d(s2, 2)), (s0, s1, s2)

    def encode(self, x):
        """Input -> bottleneck features (B,512,H/8,W/8)."""
        return self.encode_skips(x)[0]

    def decode(self, b, skips=None):
        """Bottleneck features (with context) -> full-resolution features (B,64,H,W). `skips` from `encode_skips` (needed
        when the network was built with skips=True)."""
        x = self.rb4(self.context(b))
        if not self.skips:
            x = self.rb5(self._up(x))
            x = self.rb6(self._up(x))
            return self._up(x)
        s0, s1, s2 = skips
        x = self.rb5(torch.cat([self._up(x), s2], dim=1))
        x = self.rb6(torch.cat([self._up(x), s1], dim=1))
        return self.fuse(torch.cat([self._up(x), s0], dim=1))

    def features(self, x):
        b, skips = self.encode_skips(x)
        return self.decode(b, skips)

    @staticmethod
    def flip(x):
        """180 degree rotation of the (square, zero-padded) image."""
        return torch.flip(x, dims=(2, 3))

    def initial_hidden(self, batch, canvas, device=None):
        """Empty belief for inputs of size `canvas` = (H, W)."""
        return torch.zeros(batch, self.hidden, canvas[0] // 8, canvas[1] // 8, device=device)

    def update_belief(self, b0, h_prev, a_prev):
        """Latent belief update from the encoded observation (rotation 0) and the encoded last action.
        Returns (new hidden state, context to add to the bottleneck features)."""
        h = self.gru(self.rec_in(torch.cat([b0, a_prev], 1)), h_prev)
        return h, self.rec_out(h)

    def _with_held(self, b, held):
        """Add the embedding of the held class (zeros = nothing held) to the bottleneck features."""
        return b if held is None or not self.learned_placement else b + self.held_embed(held)[:, :, None, None]

    def _features(self, x, h_prev, a_prev, held):
        """Decoder features of the input (f0) and of its 180 degree rotation (f1), and the new belief state."""
        both, both_skips = self.encode_skips(torch.cat([x, self.flip(x)]))  # both push directions in one batched pass
        b0, b1 = (self._with_held(b, held) for b in both.chunk(2))
        skips0 = tuple(s.chunk(2)[0] for s in both_skips)
        skips1 = tuple(s.chunk(2)[1] for s in both_skips)
        h = None
        if self.belief == 'recurrent':
            h, ctx = self.update_belief(b0, h_prev, a_prev)
            b0, b1 = b0 + ctx, b1 + self.flip(ctx)
        return self.decode(b0, skips0), self.decode(b1, skips1), h

    @staticmethod
    def _pooled(f):
        return torch.cat([f.mean(dim=(2, 3)), f.amax(dim=(2, 3))], dim=1)

    def forward_ext(self, x, h_prev=None, a_prev=None, held=None):
        """All outputs. x: (B,C,H,W) -> `Outputs`: push (B,2L,H,W) [index = direction*L + length], pick (B,n_pick,H,W), place
        (B,1,H,W or None) in input orientation, the new hidden state (None unless belief == 'recurrent') and, for the
        hierarchical network, the push length proposed by the actor (B,2,H,W; 0..1 of the length range) and the value of
        every primitive (B,3: push, pick, place); push then is the critic evaluated at the proposed length.
        held: (B,3) one-hot class of the object in the gripper (zeros when nothing is held)."""
        f0, f1, h = self._features(x, h_prev, a_prev, held)
        place = self.place_head(f0) if self.learned_placement else None
        if not self.hierarchical:
            push = torch.cat([self.push_head(f0), self.flip(self.push_head(f1))], dim=1)
            return Outputs(push, self.pick_head(f0), place, h, None, None)
        l0, l1 = torch.sigmoid(self.length_head(f0)), torch.sigmoid(self.length_head(f1))
        push = torch.cat([self._push_critic(f0, l0), self.flip(self._push_critic(f1, l1))], dim=1)
        lengths = torch.cat([l0, self.flip(l1)], dim=1)
        return Outputs(push, self.pick_head(f0), place, h, lengths, self.prim_head(self._pooled(f0.detach())))

    def forward(self, x, h_prev=None, a_prev=None, held=None):
        """(push, pick, place, hidden) of `forward_ext`."""
        out = self.forward_ext(x, h_prev, a_prev, held)
        return out.push, out.pick, out.place, out.hidden

    def _push_critic(self, f, length):
        return self.push_head(torch.cat([f, length], dim=1))

    def forward_train(self, x, length, h_prev=None, a_prev=None, held=None):
        """Training outputs of the hierarchical network, from one pass.

        length: (B,) executed push length (0..1). Returns a dict: `push_exec` (B,2,H,W) critic at the *executed* length
        (for the TD loss), `pick`, `place`, `push_actor` (B,2,H,W) critic at the actor's length with detached features
        (its gradient reaches the actor only: DDPG), `lengths`, `prim` (B,3) value of every primitive."""
        f0, f1, _ = self._features(x, h_prev, a_prev, held)
        fixed = length[:, None, None, None].expand(-1, 1, *f0.shape[2:])
        push_exec = torch.cat([self._push_critic(f0, fixed), self.flip(self._push_critic(f1, fixed))], dim=1)
        l0, l1 = torch.sigmoid(self.length_head(f0.detach())), torch.sigmoid(self.length_head(f1.detach()))
        push_actor = torch.cat([self._push_critic(f0.detach(), l0), self.flip(self._push_critic(f1.detach(), l1))], dim=1)
        return {'push_exec': push_exec, 'pick': self.pick_head(f0), 'push_actor': push_actor,
                'place': self.place_head(f0) if self.learned_placement else None,
                'lengths': torch.cat([l0, self.flip(l1)], dim=1), 'prim': self.prim_head(self._pooled(f0.detach()))}

    PICK, PUSH, PLACE = 0, 1, 2

    def forward_selected(self, x, primitive, rotation, h_prev=None, a_prev=None, held=None):
        """Training path: the Q-map of the taken primitive only, in input orientation: (B,1,H,W).
        primitive (B,) long in {PICK, PUSH, PLACE}; rotation (B,) long = push index direction*L + length (pushes), jaw yaw index (gripper picks)."""
        is_push = primitive == self.PUSH
        L = self.n_push_lengths
        fl = (is_push & (rotation // L == 1))[:, None, None, None]
        b, skips = self.encode_skips(torch.where(fl, self.flip(x), x))
        b = self._with_held(b, held)
        if self.belief == 'recurrent':
            b0 = self._with_held(self.encode(x), held)  # belief is always computed in the un-rotated frame
            _, ctx = self.update_belief(b0, h_prev, a_prev)
            b = b + torch.where(fl, self.flip(ctx), ctx)
        f = self.decode(b, skips)
        pr = primitive[:, None, None, None]
        push_maps = self.push_head(f)  # (B,L,H,W): pick the channel of the chosen length
        push_out = push_maps.gather(1, (rotation % L)[:, None, None, None].expand(-1, 1, *push_maps.shape[2:]))
        pick_maps = self.pick_head(f)  # (B,n_pick,H,W): for the gripper the channel is the jaw yaw (the 'rotation' of a pick)
        pick_out = pick_maps.gather(1, (rotation * (primitive == self.PICK)).clamp(max=self.n_pick - 1)[:, None, None, None]
                                    .expand(-1, 1, *pick_maps.shape[2:]))
        other = torch.where(pr == self.PLACE, self.place_head(f), pick_out) if self.learned_placement else pick_out
        out = torch.where(pr == self.PUSH, push_out, other)
        return torch.where(fl, self.flip(out), out)
