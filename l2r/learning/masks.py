"""Structural feasibility masks for the dense Q-maps (pure numpy, from the height map alone).

`basic` (the original): grasp on any object cell, push on cells near an object (a dilated ring).
`structured` removes actions that are known to be useless from how the primitives work:

* grasp: only the cells just inside the FRONT face of an object (the suction approach comes from the front and stops on contact, so
  deeper cells repeat the same contact; the front-most cell itself often misses by a millimetre, see demos.GRASP_DEPTH_CELLS);
* push: only start cells from which the blade, moving in that direction, has something in its path: object cells in the strip
  of columns the blade sweeps (+-2 cells), per direction and length, vectorised. (`rearrange.check_push` is looser: it counts
  objects in every column from the start cell to the front.)
"""
import numpy as np
from scipy import ndimage

OBJECT_HEIGHT = 0.008  # m: height map cells above this belong to an object (same as the basic masks)
PUSH_OBJECT_HEIGHT = 0.02  # m: `check_push` counts cells above this
MIN_PUSH_CELLS = 3  # `check_push` requires more than 2 object cells in the blade's path
BLADE_HALF_WIDTH_CELLS = 2  # the tip has a 1 cm radius = 2 cells of 5 mm: the strip the blade sweeps is +-2 cells around its column


GRASP_EDGE_MARGIN = 0.15  # fraction of an object's width at each side that is excluded (measured: suction grasps within ~15% of
                          # the edge fail far more often: scripts/grasp_profile.py)


def front_grasp_cells(object_cells, depth_cells=(1, 3), edge_margin=GRASP_EDGE_MARGIN):
    """Cells within `depth_cells` (min, max) behind the front-most cell of every run of object cells along x (columns).

    Larger column = nearer the opening, so a run's front is its largest column. Adjacent objects in one row form one run, which
    keeps only the front-most of them: the others are occluded from the cup anyway.

    edge_margin: the cells in the outermost `edge_margin` of each object's width (its row extent) are dropped: a cup on the edge of a
    curved or irregular object does not seal. 0 keeps the whole front band."""
    mask = np.zeros_like(object_cells, dtype=bool)
    lo, hi = depth_cells
    for row in range(object_cells.shape[0]):
        cells = object_cells[row]
        if not cells.any():
            continue
        padded = np.concatenate([[False], cells, [False]])
        starts, ends = np.flatnonzero(~padded[:-1] & padded[1:]), np.flatnonzero(padded[:-1] & ~padded[1:]) - 1
        for start, end in zip(starts, ends):
            first, last = max(end - hi, start), end - lo
            if last < first:  # a run too short to have cells that deep: its last cell
                first = last = end
            mask[row, first:last + 1] = True
    if edge_margin > 0:
        labels, _ = ndimage.label(object_cells, structure=np.ones((3, 3)))
        for k, extent in enumerate(ndimage.find_objects(labels), 1):
            first, last = extent[0].start, extent[0].stop - 1
            if last == first:
                continue
            for row in range(first, last + 1):
                if not first + edge_margin * (last - first) - 1e-9 <= row <= last - edge_margin * (last - first) + 1e-9:
                    mask[row, labels[row] == k] = False
    return mask


def push_path_cells(objects, length_cells, towards_increasing_rows, half_width=None):
    """Cells (row, col) from which a push of `length_cells` rows has at least MIN_PUSH_CELLS object cells in its path.

    half_width=None: the rectangle of rows [row, row + length] (or [row - length, row]) and ALL columns >= col, which is what
    `check_push` inspects (loose: an object beside the blade counts). half_width=b: only the strip of columns [col - b, col + b],
    the region the blade really sweeps."""
    h, w = objects.shape
    if half_width is None:
        per_row = np.cumsum(objects[:, ::-1], axis=1)[:, ::-1].astype(np.int32)  # objects in columns >= col, per row
    else:
        prefix = np.concatenate([np.zeros((h, 1), np.int32), np.cumsum(objects, axis=1, dtype=np.int32)], axis=1)
        cols = np.arange(w)
        per_row = prefix[:, np.minimum(cols + half_width + 1, w)] - prefix[:, np.maximum(cols - half_width, 0)]
    stacked = np.concatenate([np.zeros((1, w), np.int32), np.cumsum(per_row, axis=0)])  # stacked[r] = rows < r
    rows = np.arange(h)
    if towards_increasing_rows:
        low, high = rows, np.minimum(rows + length_cells, h - 1)
    else:
        low, high = np.maximum(rows - length_cells, 0), rows
    return (stacked[high + 1] - stacked[low]) >= MIN_PUSH_CELLS


def structured_masks(crop, push_lengths_px, edge_margin=GRASP_EDGE_MARGIN):
    """(grasp (h,w) bool, push (2L,h,w) bool) from the cropped height map. Push index = direction * L + length; direction 0 moves
    towards decreasing rows (angle -90 degrees), direction 1 towards increasing rows."""
    grasp = front_grasp_cells(crop > OBJECT_HEIGHT, edge_margin=edge_margin)
    near = crop > PUSH_OBJECT_HEIGHT
    push = [push_path_cells(near, k, direction == 1, BLADE_HALF_WIDTH_CELLS) & ~(crop > OBJECT_HEIGHT)
            for direction in (0, 1) for k in push_lengths_px]
    return grasp, np.stack(push)
