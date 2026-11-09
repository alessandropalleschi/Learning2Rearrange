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


class QNet(nn.Module):
    nr_rotations = 2  # push left / right (0 and 180 degrees)

    def __init__(self, in_channels=IN_CHANNELS, belief='none', hidden=64, n_push_lengths=1, learned_placement=True,
                 backbone='conv', transformer_dim=192, transformer_layers=3):
        """learned_placement: build the place head (conditioned on the held class). Without it `forward` returns
        `None` as place map: the head would cost about as much as the pick head for nothing."""
        super().__init__()
        self.belief, self.hidden, self.n_push_lengths = belief, hidden, n_push_lengths
        self.learned_placement, self.backbone = learned_placement, backbone
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
        self.rb4, self.rb5, self.rb6 = ResidualBlock(512, 256), ResidualBlock(256, 128), ResidualBlock(128, 64)
        # one Q-map per selectable push length; the second direction is the same head on the flipped input
        self.push_head = nn.Sequential(ResidualBlock(64, 64), nn.Conv2d(64, n_push_lengths, 1))
        self.pick_head = nn.Sequential(ResidualBlock(64, 64), nn.Conv2d(64, 1, 1))
        if learned_placement:  # Q-map over the free cells, conditioned on the class of the object in the gripper
            self.place_head = nn.Sequential(ResidualBlock(64, 64), nn.Conv2d(64, 1, 1))
            self.held_embed = nn.Linear(len(HELD_CLASSES), 512)
            nn.init.zeros_(self.held_embed.weight)
            nn.init.zeros_(self.held_embed.bias)

    @staticmethod
    def _up(x):
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

    def encode(self, x):
        """Input -> bottleneck features (B,512,H/8,W/8)."""
        x = self.stem(x)
        x = self.rb1(F.max_pool2d(x, 2))
        x = self.rb2(F.max_pool2d(x, 2))
        return self.rb3(F.max_pool2d(x, 2))

    def decode(self, b):
        """Bottleneck features (with context) -> full-resolution features (B,64,H,W)."""
        x = self.rb4(self.context(b))
        x = self.rb5(self._up(x))
        x = self.rb6(self._up(x))
        return self._up(x)

    def features(self, x):
        return self.decode(self.encode(x))

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

    def forward(self, x, h_prev=None, a_prev=None, held=None):
        """All Q-maps. x: (B,C,H,W) -> push (B,2L,H,W) [index = direction*L + length], pick (B,1,H,W), place (B,1,H,W) in input
        orientation, and the new hidden state (None unless belief == 'recurrent').
        held: (B,3) one-hot class of the object in the gripper (zeros when nothing is held)."""
        both = self.encode(torch.cat([x, self.flip(x)]))  # both push directions in one batched pass
        b0, b1 = (self._with_held(b, held) for b in both.chunk(2))
        h = None
        if self.belief == 'recurrent':
            h, ctx = self.update_belief(b0, h_prev, a_prev)
            b0, b1 = b0 + ctx, b1 + self.flip(ctx)
        f0, f1 = self.decode(b0), self.decode(b1)
        push = torch.cat([self.push_head(f0), self.flip(self.push_head(f1))], dim=1)
        return push, self.pick_head(f0), self.place_head(f0) if self.learned_placement else None, h

    PICK, PUSH, PLACE = 0, 1, 2

    def forward_selected(self, x, primitive, rotation, h_prev=None, a_prev=None, held=None):
        """Training path: the Q-map of the taken primitive only, in input orientation: (B,1,H,W).
        primitive (B,) long in {PICK, PUSH, PLACE}; rotation (B,) long = push index direction*L + length (pushes only)."""
        is_push = primitive == self.PUSH
        L = self.n_push_lengths
        fl = (is_push & (rotation // L == 1))[:, None, None, None]
        b = self._with_held(self.encode(torch.where(fl, self.flip(x), x)), held)
        if self.belief == 'recurrent':
            b0 = self._with_held(self.encode(x), held)  # belief is always computed in the un-rotated frame
            _, ctx = self.update_belief(b0, h_prev, a_prev)
            b = b + torch.where(fl, self.flip(ctx), ctx)
        f = self.decode(b)
        pr = primitive[:, None, None, None]
        push_maps = self.push_head(f)  # (B,L,H,W): pick the channel of the chosen length
        push_out = push_maps.gather(1, (rotation % L)[:, None, None, None].expand(-1, 1, *push_maps.shape[2:]))
        other = torch.where(pr == self.PLACE, self.place_head(f), self.pick_head(f)) if self.learned_placement \
            else self.pick_head(f)
        out = torch.where(pr == self.PUSH, push_out, other)
        return torch.where(fl, self.flip(out), out)
