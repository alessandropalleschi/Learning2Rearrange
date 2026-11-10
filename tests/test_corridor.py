"""Corridor preference of the heuristic placement never trades the object's region for a clear corridor."""
from types import SimpleNamespace

import numpy as np

from l2r.envs.rearrange import RearrangeEnv, corridor_choice

SHAPE = (20, 30)


def masks():
    free = np.ones(SHAPE, np.float32)
    own = np.zeros(SHAPE, bool)
    own[5:10] = True
    return free, own


def test_case_1_clear_cells_inside_the_region():
    free, own = masks()
    kept = np.zeros(SHAPE, bool)
    kept[6, 3:8] = True  # clear inside
    kept[15, :] = True  # clear outside
    out = corridor_choice(free, kept, own)
    assert out.dtype == free.dtype and out.sum() == 5 and out[6, 3:8].all()


def test_case_2_region_has_room_but_no_clear_corridor():
    free, own = masks()
    kept = np.zeros(SHAPE, bool)
    kept[15, :] = True  # only outside the region
    out = corridor_choice(free, kept, own)
    assert out is free  # the unfiltered free space: the placement stays in the region


def test_case_3_region_full_takes_a_clear_cell_elsewhere():
    free, own = masks()
    free[5:10] = 0  # no free cell in the region
    kept = np.zeros(SHAPE, bool)
    kept[15, 2:4] = True
    out = corridor_choice(free, kept, own)
    assert out.dtype == free.dtype and out.sum() == 2 and out[15, 2:4].all()


def test_case_4_nothing_clear_and_region_full():
    free, own = masks()
    free[5:10] = 0
    out = corridor_choice(free, np.zeros(SHAPE, bool), own)
    assert out is free


def test_flag_off_returns_the_input_unchanged():
    env = SimpleNamespace(cfg=SimpleNamespace(place_corridor=False))
    free = np.ones(SHAPE, np.float32)
    assert RearrangeEnv._corridor_free(env, free, np.zeros((28, 34)), (8, 8), 1) is free


def test_band_region_from_target_pixels_with_unordered_bounds():
    from l2r.sim.constants import LABELS, TARGET_ORDER
    sim = SimpleNamespace(scene=SimpleNamespace(layout='bands'), target_pixels=lambda: ([12, 2, 20], [5, 9, 25], None))
    env = SimpleNamespace(sim=sim)
    label = LABELS[TARGET_ORDER[0]]
    own = RearrangeEnv._own_region(env, label, SHAPE)
    assert own[5:12].all() and not own[:5].any() and not own[12:].any()  # rows 5..11 whichever bound is the larger
