"""Q-network: one shared fully-convolutional backbone with a push head and a pick head.

Improvements over the original two independent networks:
  * 4-channel input (RGB + height), as described in the paper (was 6 channels, depth repeated 3x).
  * GroupNorm instead of BatchNorm: statistics do not depend on the (tiny, correlated) batch.
  * Global-context block at the bottleneck: pooled features of the whole cabinet are fed back to every
    pixel, so each Q value can depend on the full scene (fully convolutional nets only see a local window).
  * Shared backbone: the push (rotation 0) and pick Q-maps reuse the same features of the un-rotated
    input, so one forward pass serves both primitives.

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


class QNet(nn.Module):
    nr_rotations = 2  # push left / right (0 and 180 degrees)

    def __init__(self, in_channels=IN_CHANNELS):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_channels, 64, 3, padding=1, bias=False), norm(64), nn.ReLU(inplace=True))
        self.rb1, self.rb2, self.rb3 = ResidualBlock(64, 128), ResidualBlock(128, 256), ResidualBlock(256, 512)
        self.context = GlobalContext(512)
        self.rb4, self.rb5, self.rb6 = ResidualBlock(512, 256), ResidualBlock(256, 128), ResidualBlock(128, 64)
        self.push_head = nn.Sequential(ResidualBlock(64, 64), nn.Conv2d(64, 1, 1))
        self.pick_head = nn.Sequential(ResidualBlock(64, 64), nn.Conv2d(64, 1, 1))

    @staticmethod
    def _up(x):
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

    def features(self, x):
        x = self.stem(x)
        x = self.rb1(F.max_pool2d(x, 2))
        x = self.rb2(F.max_pool2d(x, 2))
        x = self.rb3(F.max_pool2d(x, 2))
        x = self.context(x)
        x = self.rb4(x)
        x = self.rb5(self._up(x))
        x = self.rb6(self._up(x))
        return self._up(x)

    @staticmethod
    def flip(x):
        """180 degree rotation of the (square, zero-padded) image."""
        return torch.flip(x, dims=(2, 3))

    def forward(self, x):
        """All Q-maps. x: (B,4,H,W) -> push (B,2,H,W) [0deg,180deg], pick (B,1,H,W). Maps are in input orientation."""
        f0 = self.features(x)
        f1 = self.features(self.flip(x))
        push = torch.cat([self.push_head(f0), self.flip(self.push_head(f1))], dim=1)
        return push, self.pick_head(f0)

    def forward_selected(self, x, is_push, rotation):
        """Training path: one backbone pass per sample on the input actually needed.
        is_push (B,) bool, rotation (B,) long in {0,1} (only used where is_push).
        Returns the Q-map of the taken primitive, in input orientation: (B,1,H,W)."""
        flip = is_push & (rotation == 1)
        x = torch.where(flip[:, None, None, None], self.flip(x), x)
        f = self.features(x)
        out = torch.where(is_push[:, None, None, None], self.push_head(f), self.pick_head(f))
        return torch.where(flip[:, None, None, None], self.flip(out), out)
