"""Structured action masks: front-face grasps and directional pushes (checked against the environment's own push check)."""
import numpy as np

from l2r.config import Config
from l2r.envs.rearrange import check_push
from l2r.learning.masks import front_grasp_cells, push_path_cells
from l2r.learning.trainer import Trainer
from l2r.sim.workspace import Workspace


def test_grasp_mask_keeps_the_cells_just_behind_the_front_face_of_each_run():
    objects = np.zeros((4, 12), bool)
    objects[0, 2:9] = objects[1, 0:4] = objects[1, 6:8] = objects[2, 5] = objects[3, 4:6] = True
    mask = front_grasp_cells(objects, edge_margin=0)
    assert [np.flatnonzero(mask[r]).tolist() for r in range(4)] == [[5, 6, 7], [0, 1, 2, 6], [5], [4]]
    assert not (mask & ~objects).any()  # never outside an object


def test_push_mask_matches_the_environment_check_push_on_random_height_maps():
    """check_push (rectangle of rows from the start to the end of the push, columns from the start cell on, more than 2 cells above
    2 cm) is what the environment executes; the vectorised mask must agree wherever the push stays inside the map."""
    rng = np.random.default_rng(0)
    ws = Workspace()
    rows, cols, length_cells = 70, 60, 16
    for _ in range(4):
        depth = np.zeros((rows + 8, cols + 4), np.float32)  # padded like the observation: check_push crops 4 px on both row sides
        for _ in range(rng.integers(2, 6)):
            r, c = rng.integers(0, rows - 14), rng.integers(0, cols - 14)
            depth[4 + r:4 + r + rng.integers(6, 14), 4 + c:4 + c + rng.integers(6, 14)] = 0.05
        crop = depth[4:-4, 4:]
        for direction, towards_increasing in ((1, True), (0, False)):
            mask = push_path_cells(crop > 0.02, length_cells, towards_increasing)  # legacy semantics: half_width=None
            angle = np.deg2rad(direction * 180.0 - 90)
            for r, c in zip(rng.integers(length_cells + 1, rows - length_cells - 1, 120), rng.integers(0, cols, 120)):
                assert check_push(ws, depth, int(c), int(r), angle, length_cells * ws.pixel_size) == bool(mask[r, c]), (direction, r, c)


def test_structured_masks_through_the_trainer():
    depth = np.zeros((128, 104), np.float32)
    depth[50:70, 40:70] = 0.06  # one object, 20 rows (y) x 30 columns (x)
    basic = Trainer(Config(mask_level='basic'))
    structured = Trainer(Config(mask_level='structured'))
    assert structured.valid_mask(depth, 'grasp').sum() < basic.valid_mask(depth, 'grasp').sum() / 5  # a thin front band
    assert structured.valid_mask(depth, 'push').shape == basic.valid_mask(depth, 'push').shape == (2, 120, 100)
    push_s, push_b = structured.valid_mask(depth, 'push'), basic.valid_mask(depth, 'push')
    assert push_s.sum() > 0
    objects = (depth[4:-4, 4:] > 0.008)
    assert not (push_s & objects[None]).any()  # the blade never starts inside an object
    # direction 1 pushes towards increasing rows: its start cells are on the low-row side of the object, direction 0 on the other
    assert np.nonzero(push_s[1].any(axis=1))[0].mean() < np.nonzero(push_s[0].any(axis=1))[0].mean()


def test_blade_strip_mask_requires_an_object_in_the_blades_own_columns():
    objects = np.zeros((40, 30), bool)
    objects[20:26, 10:14] = True
    towards_up = push_path_cells(objects, 10, True, half_width=2)  # moving towards increasing rows
    assert towards_up[12, 10] and towards_up[12, 8] and towards_up[12, 15]  # within +-2 columns of the object, 8 rows before it
    assert not towards_up[12, 5] and not towards_up[12, 20]  # an object beside the blade is not in its path
    assert not towards_up[30, 11] and not towards_up[5, 11]  # the object is behind the blade / too far ahead
    loose = push_path_cells(objects, 10, True)
    assert loose[12, 5] and not towards_up[12, 5]  # check_push's rectangle accepts what the strip rejects
    towards_down = push_path_cells(objects, 10, False, half_width=2)
    assert towards_down[30, 11] and not towards_down[12, 11]


def test_grasp_mask_drops_the_edges_of_an_objects_width():
    objects = np.zeros((20, 30), bool)
    objects[2:12, 10:24] = True  # one object, 10 rows wide
    full = front_grasp_cells(objects, edge_margin=0)
    centre = front_grasp_cells(objects, edge_margin=0.15)
    rows_full, rows_centre = sorted(set(np.nonzero(full)[0])), sorted(set(np.nonzero(centre)[0]))
    assert rows_full == list(range(2, 12))
    assert rows_centre == list(range(4, 10))  # rows within 15% of the 2..11 extent (1.35 rows) are dropped on both sides
    assert (centre <= full).all() and centre.sum() == 6 * 3
    two = np.zeros((20, 30), bool)
    two[1:5, 10:20] = two[12:18, 10:20] = True  # separate objects: each gets its own margin
    rows = sorted(set(np.nonzero(front_grasp_cells(two, edge_margin=0.3))[0]))
    assert rows == [2, 3, 14, 15]  # objects span rows 1..4 and 12..17: boundaries at 1.9 / 3.1 and 13.5 / 15.5
