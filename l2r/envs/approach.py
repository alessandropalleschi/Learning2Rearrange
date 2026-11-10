"""Collision-aware approach lines for grasps, from the height map alone (no privileged information).

A grasp comes in along a straight line from the cabinet front to the target and leaves along the same line with the object. Anything
raised in that corridor (other objects) is in the way: the tool or the held object knocks it over. `choose_approach` scores candidate
approach angles by the number of raised cells of OTHER objects in the swept corridor and returns the clearest one (the straight-in
approach on ties). Coordinates: action/heightmap cells (row = y index, col = x index; the robot is beyond the last column)."""
import numpy as np

from l2r.learning.object_centric import crop, extract_instances

RAISED = 0.01  # m: a cell higher than this is an obstacle
TOOL_HALF_WIDTH_CELLS = 4  # half width of the tool (cup / jaws) in cells (5 mm)
MARGIN_CELLS = 2
CANDIDATES = np.deg2rad([0, 15, -15, 30, -30, 45, -45])


def corridor_mask(shape, target_rc, approach, half_width):
    """Cells within `half_width` of the approach line from the target out to the front (col -> max), plus the target's surroundings."""
    rows, cols = np.mgrid[0:shape[0], 0:shape[1]]
    direction = np.array([-np.sin(approach), -np.cos(approach)])  # (d row, d col) of the motion INTO the cabinet
    v_row, v_col = rows - target_rc[0], cols - target_rc[1]
    towards_front = -(v_row * direction[0] + v_col * direction[1])  # distance from the target back along the line, in cells
    across = np.abs(v_row * direction[1] - v_col * direction[0])
    return (towards_front >= -MARGIN_CELLS) & (across <= half_width)


def entry_row(shape, target_rc, approach):
    """Row where the approach line crosses the front edge (last column)."""
    return target_rc[0] + (shape[1] - 1 - target_rc[1]) * np.tan(approach)


def own_cells(obs, target_rc):
    """Cells of the object at the target (its instance from the observation), or a small disc if none is found there."""
    for inst in extract_instances(obs):
        if np.any((np.abs(inst.rows - target_rc[0]) <= 1) & (np.abs(inst.cols - target_rc[1]) <= 1)):
            return inst.rows, inst.cols
    return np.array([target_rc[0]]), np.array([target_rc[1]])


def obstacles(obs, target_rc, approach):
    """Number of raised cells of other objects in the corridor swept by the tool and the held object for this approach."""
    depth = crop(obs['depthmap'])
    rows, cols = own_cells(obs, target_rc)
    own = np.zeros(depth.shape, bool)
    own[rows, cols] = True
    across = np.array([np.cos(approach), -np.sin(approach)])  # (row, col) unit vector perpendicular to the approach
    extent = (rows - target_rc[0]) * across[0] + (cols - target_rc[1]) * across[1]
    half_width = max(TOOL_HALF_WIDTH_CELLS, np.abs(extent).max() if extent.size else 0) + MARGIN_CELLS
    corridor = corridor_mask(depth.shape, target_rc, approach, half_width)
    return int(np.sum(corridor & (depth > RAISED) & ~own))


def choose_approach(obs, target_rc, candidates=CANDIDATES, opening_margin_cells=6):
    """(approach angle, obstacle count) of the clearest feasible candidate; ties go to the earlier candidate (straight in first)."""
    shape = crop(obs['depthmap']).shape
    best = None
    for approach in candidates:
        row = entry_row(shape, target_rc, approach)
        if not opening_margin_cells <= row <= shape[0] - 1 - opening_margin_cells:
            continue  # the line would enter through a side wall
        count = obstacles(obs, target_rc, approach)
        if best is None or count < best[1]:
            best = (float(approach), count)
    return best if best is not None else (0.0, obstacles(obs, target_rc, 0.0))
