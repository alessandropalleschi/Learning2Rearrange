"""A small U-Net for semantic segmentation of the object types (CPU friendly: ~0.5 M parameters)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.GroupNorm(min(8, cout), cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.GroupNorm(min(8, cout), cout), nn.ReLU(inplace=True))


class SegmentationNet(nn.Module):
    """Input: RGB in [0, 1] + depth / 1.5 m (4 channels), H and W multiples of 8.
    Output: logits (B, 1 + number of object types, H, W); channel 0 = background."""

    def __init__(self, num_types, in_channels=4, widths=(16, 32, 64, 96)):
        super().__init__()
        self.num_types = num_types
        self.enc = nn.ModuleList([block(in_channels, widths[0])] + [block(a, b) for a, b in zip(widths, widths[1:])])
        self.dec = nn.ModuleList([block(b + a, a) for a, b in zip(widths, widths[1:])])
        self.head = nn.Conv2d(widths[0], 1 + num_types, 1)

    def forward(self, x):
        skips = []
        for i, enc in enumerate(self.enc):
            x = enc(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        for dec, skip in zip(reversed(self.dec), reversed(skips[:-1])):
            x = dec(torch.cat([skip, F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)], 1))
        return self.head(x)
