"""Spatial belief: a persistent per-episode memory of what was seen in the cabinet.

The side camera cannot see behind objects; those cells are grey ("occluded") in the colormap. The tracker
remembers, for every cell, the last observed colour / height / object id and how many steps ago that was,
plus where the last action happened. It is exposed to the network as extra input channels, so a
fully-convolutional policy can reason about objects that are currently hidden.

Channels of the compact (uint8) representation, BELIEF_CHANNELS = 7:
    0-2  remembered RGB            3  remembered height (0..0.4 m)      4  age (0 = seen now, 255 = stale/never seen)
    5    last push location        6  last pick (255) / place (128) location
"""
import numpy as np

BELIEF_CHANNELS = 7
OCCLUDED = np.array([105, 105, 105], np.uint8)


class BeliefTracker:
    def __init__(self, age_horizon=10, offset_px=4):
        self.age_horizon = age_horizon
        self.offset_px = offset_px
        self.state = None

    def reset(self):
        self.state = None

    def update(self, obs, last_action=None):
        """obs: full observation (colormap, depthmap, idmap). last_action: {'primitive','pix','place_pix'} or None.
        Returns the belief dict {'rgb','height','id','age','observed','push','pick'} (float arrays, H x W)."""
        color, depth = obs['colormap'], obs['depthmap']
        h, w = depth.shape[:2]
        if self.state is None:
            self.state = {'rgb': np.full((h, w, 3), OCCLUDED, np.uint8), 'height': np.zeros((h, w), np.float32),
                          'id': np.zeros((h, w), np.float32),
                          'age': np.full((h, w), self.age_horizon, np.float32)}
        s = self.state
        observed = ~np.all(color == OCCLUDED, axis=-1)
        s['rgb'][observed] = color[observed]
        s['height'][observed] = depth[observed]
        s['id'][observed] = obs['idmap'][observed]
        s['age'] = np.where(observed, 0.0, np.minimum(s['age'] + 1, self.age_horizon)).astype(np.float32)

        push = np.zeros((h, w), np.float32)
        pick = np.zeros((h, w), np.float32)
        if last_action is not None:
            if last_action['primitive'] == 'place':
                self._mark(pick, last_action['pix'], 0.5)
            else:
                self._mark(push if last_action['primitive'] == 'push' else pick, last_action['pix'], 1.0)
            if last_action.get('place_pix') is not None:
                px, py = last_action['place_pix']  # (col, row) in action coordinates
                self._mark(pick, (0, py, px), 0.5)
        return dict(s, observed=observed, push=push, pick=pick,
                    rgb=s['rgb'].copy(), height=s['height'].copy(), id=s['id'].copy(), age=s['age'].copy())

    def _mark(self, channel, pix, value, radius=2):
        r, c = int(pix[1]) + self.offset_px, int(pix[2]) + self.offset_px
        channel[max(r - radius, 0):r + radius + 1, max(c - radius, 0):c + radius + 1] = value


def compact_belief(b, age_horizon=10):
    """Belief dict -> (7,H,W) uint8 array (see module docstring)."""
    out = np.zeros((BELIEF_CHANNELS,) + b['height'].shape, np.uint8)
    out[0:3] = np.transpose(b['rgb'], (2, 0, 1))
    out[3] = np.clip(b['height'] / 0.4, 0, 1) * 255
    out[4] = np.clip(b['age'] / age_horizon, 0, 1) * 255
    out[5] = b['push'] * 255
    out[6] = b['pick'] * 255
    return out
