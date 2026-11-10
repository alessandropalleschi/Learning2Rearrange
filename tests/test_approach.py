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


def test_place_corridor_counts_other_objects_between_a_cell_and_the_front():
    from l2r.envs.approach import place_corridor_obstacles
    depth = np.zeros((120, 100), np.float32)
    depth[50:60, 70:80] = 0.1  # an object between the back-middle and the front
    counts = place_corridor_obstacles(depth, 4)
    assert counts[55, 30] == 90  # behind it: rows 51..59 of the object (half width 4) are in the corridor
    assert counts[55, 90] == 0  # in front of it: nothing to pass
    assert counts[20, 30] == 0  # other rows: clear
    assert 0 < counts[47, 30] < 100  # the corridor edge clips it


def test_goal_masks_paint_only_free_cells():
    from l2r.sim.vision import top_down_maps
    from l2r.sim.workspace import Workspace
    ws = Workspace()
    h, w = ws.heightmap_shape
    b = ws.bounds
    xs, ys = np.meshgrid(np.linspace(b[0][0] + 0.003, b[0][1] - 0.003, w), np.linspace(b[1][0] + 0.003, b[1][1] - 0.003, h))
    points = np.stack([xs.ravel(), ys.ravel(), np.full(xs.size, b[2][0])], 1)
    colors = np.full((points.shape[0], 3), 255, np.uint8)
    mask = np.zeros((h, w), bool)
    mask[10:20, 30:40] = True
    _, colormap = top_down_maps(points, colors, ws, ('masks', [mask, np.zeros_like(mask), np.zeros_like(mask)], [(0, 0, 139), (0, 0, 0), (0, 0, 0)]))
    assert (colormap[mask] == (0, 0, 139)).all() and (colormap[~mask] == 255).all()
