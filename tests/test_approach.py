"""Collision-aware approach lines from the height map."""
import numpy as np

from l2r.envs.approach import choose_approach, entry_row, obstacles
from l2r.learning.object_centric import CLASS_RGB

H, W = 128, 104


def scene(blocker=True):
    """A blue object at rows 55..64, cols 30..39 (cropped coordinates) and, optionally, a green one straight in front of it."""
    color = np.full((H, W, 3), 255, np.uint8)
    depth = np.zeros((H, W), np.float32)
    color[4 + 55:4 + 65, 4 + 30:4 + 40] = CLASS_RGB[0]
    depth[4 + 55:4 + 65, 4 + 30:4 + 40] = 0.1
    if blocker:
        color[4 + 54:4 + 66, 4 + 60:4 + 70] = CLASS_RGB[1]
        depth[4 + 54:4 + 66, 4 + 60:4 + 70] = 0.1
    return {'colormap': color, 'depthmap': depth, 'regions': np.array([[10, 30], [70, 90], [100, 115]])}


def test_a_blocker_in_front_makes_the_straight_approach_bad_and_an_angled_one_is_chosen():
    target = (60, 38)  # the front of the blue object
    assert obstacles(scene(False), target, 0.0) == 0  # its own cells do not count
    assert obstacles(scene(), target, 0.0) > 50
    approach, count = choose_approach(scene(), target)
    assert count < obstacles(scene(), target, 0.0) / 4 and abs(approach) >= np.deg2rad(15)
    assert choose_approach(scene(False), target) == (0.0, 0)  # nothing in the way: straight in


def test_entry_row_follows_the_approach_angle():
    assert entry_row((120, 100), (60, 40), 0.0) == 60
    assert entry_row((120, 100), (60, 40), np.deg2rad(45)) > 60  # positive angle: comes in from larger rows (larger y)
