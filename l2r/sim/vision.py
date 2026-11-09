"""From the camera image to the maps the policy sees: point cloud -> top-down heightmap / colormap, per-object
class and id maps, and the free-space map used by the placement heuristic. Pure numpy / OpenCV, no simulator."""
from functools import lru_cache

import numpy as np
from scipy import ndimage

from l2r.sim.constants import COLORS, LABELS, OCCLUDED_RGB, TARGET_ORDER, class_rgb255

PAD = 4  # the colormap / heightmap returned to the policy get a 4 px border (rows: both sides, columns: left)


@lru_cache(maxsize=8)
def _pixel_offsets(height, width, cx, cy):
    """(column - cx, row - cy) of every pixel; depends only on the camera, so it is computed once."""
    px, py = np.meshgrid(np.linspace(0, width - 1, width), np.linspace(0, height - 1, height))
    return px - cx, py - cy


def depth_to_points(depth, intrinsics):
    """HxW metric depth -> HxWx3 float32 points in camera coordinates."""
    dx, dy = _pixel_offsets(*depth.shape, intrinsics[0, 2], intrinsics[1, 2])
    return np.float32([dx * (depth / intrinsics[0, 0]), dy * (depth / intrinsics[1, 1]), depth]).transpose(1, 2, 0)


def camera_to_world(points, position, rotation):
    """HxWx3 camera-frame points -> world frame: rotation (3x3) and position of the camera."""
    return (points.astype(np.float64) @ np.asarray(rotation, np.float64).T
            + np.asarray(position, np.float64)).astype(np.float32)


def _project(points, values, bounds, pixel_size, inclusive):
    """Select the points inside the workspace, z-sort them (highest last = z-buffer) and compute their pixel."""
    width = int(np.round((bounds[0, 1] - bounds[0, 0]) / pixel_size))
    height = int(np.round((bounds[1, 1] - bounds[1, 0]) / pixel_size))
    compare = (lambda a, lo, hi: (a >= lo) & (a <= hi)) if inclusive else (lambda a, lo, hi: (a >= lo) & (a < hi))
    valid = np.ones(points.shape[:-1], bool)
    for axis in range(3):
        valid &= compare(points[..., axis], bounds[axis, 0], bounds[axis, 1])
    points, values = points[valid], values[valid]
    order = np.argsort(points[:, -1])
    points, values = points[order], values[order]
    px = np.clip(np.int32(np.floor((points[:, 0] - bounds[0, 0]) / pixel_size)), 0, width - 1)
    py = np.clip(np.int32(np.floor((points[:, 1] - bounds[1, 0]) / pixel_size)), 0, height - 1)
    return points, values, px, py, (height, width)


def top_down_maps(points, colors, ws, target_rows):
    """Heightmap (m above the cabinet floor, float32) and colormap (uint8; grey = occluded, white = free floor
    seen by the camera, the free part of each target region painted in its class colour)."""
    bounds = ws.bounds
    points, colors, px, py, (height, width) = _project(points, colors, bounds, ws.pixel_size, inclusive=True)
    heightmap = np.zeros((height, width), dtype=np.float32)
    colormap = np.full((height, width, colors.shape[-1]), OCCLUDED_RGB, dtype=np.uint8)
    heightmap[py, px] = points[:, 2] - bounds[2][0]
    heightmap[heightmap < 0.008] = 0  # floor noise
    for c in range(colors.shape[-1]):
        colormap[py, px, c] = colors[:, c]
    for low, high, rgb in zip(*target_rows):
        region = colormap[low:high, :width]
        region[np.all(region == 255, axis=-1)] = rgb
    return heightmap, colormap


def recolor_by_class(color, segm, object_names):
    """Recolour the camera image by object class (in place): objects in their class colour, everything else white.

    Args:
        color: HxWx3 uint8. segm: HxW int object uid per pixel. object_names: {object uid: 'class YcbName'}.
    Returns:
        color and `segm` restricted to the objects (uid of the object at every pixel, 0 for background / robot /
        cabinet). Ids are never rewritten in place, so labels can not collide with ids still to be processed.
    """
    objects = np.where(np.isin(segm, list(object_names)), segm, 0)
    color[objects == 0] = 255
    for uid in np.unique(objects[objects != 0]):
        color[objects == uid] = class_rgb255(object_names[uid].split(' ')[0])
    return color, objects


MIN_PIECE_CELLS = 8  # heightmap cells (5 mm): smaller fragments of a split instance are noise


def split_components(top):
    """Give every 8-connected piece of an instance in the top view its own id (the first piece keeps the id).

    A segmentation network sees the image, where objects in front of / behind each other overlap and merge; in the
    top view they are separated. Returns the new id map and {new id: original id}."""
    out, origin, next_id = np.zeros_like(top), {}, int(top.max()) + 1
    for uid in np.unique(top[top != 0]):
        labels, count = ndimage.label(top == uid, structure=np.ones((3, 3)))
        first = True
        for k in range(1, count + 1):
            piece = labels == k
            if piece.sum() < MIN_PIECE_CELLS:
                continue
            new = uid if first else next_id
            if not first:
                next_id += 1
            first = False
            out[piece] = new
            origin[new] = uid
    return out, origin


def object_maps(points, objects, object_names, ws, target_rows, split=False):
    """Per-object maps on the heightmap grid, from ONE projection of the object-id image: the id of the highest point
    of every cell gives the visible objects (no per-object pass over the point cloud).

    split: split instances into their pieces in the top view (for a learned perception, see `split_components`).

    Returns the id map, the class-label map, the occupancy of all visible objects, the cells of the objects that
    already stand inside the target region of their class (their centroid row is inside it) and {id: original id} of the
    split instances (empty without `split`).
    """
    shape = ws.heightmap_shape
    id_map = np.zeros(shape, np.float32)
    class_map, binary, on_target = np.zeros(shape, np.float32), np.zeros(shape, np.float32), np.zeros(shape, np.float32)
    _, ids, px, py, _ = _project(points, objects, ws.bounds, ws.pixel_size, inclusive=False)
    top = np.zeros(shape, np.int64)
    top[py, px] = ids  # points are sorted by height: the highest point of a cell is written last
    origin = {}
    if split:
        top, origin = split_components(top)
    low, high = target_rows[0], target_rows[1]
    for uid in np.unique(top[top != 0]):
        cells = top == uid
        name = object_names[origin.get(uid, uid)].split(' ')[0]
        rows = np.nonzero(cells)[0]
        centroid = int(rows.sum() / rows.size)  # centroid row of the visible part
        k = TARGET_ORDER.index(name)
        id_map[cells], class_map[cells], binary[cells] = uid, LABELS[name], 1
        if low[k] <= centroid <= high[k]:
            on_target[cells] = -1
    return id_map, class_map, binary, on_target, origin


def top_view_ids(depth, intrinsics, position, rotation, object_ids, object_names, ws):
    """Object id of the highest point of every heightmap cell for a ground-truth id image (privileged information)."""
    points = camera_to_world(depth_to_points(depth, intrinsics), position, rotation)
    objects = np.where(np.isin(object_ids, list(object_names)), object_ids, 0)
    return object_maps(points, objects, object_names, ws, ([0, 0, 0], [0, 0, 0], None))[0]


def free_space_map(binary, on_target):
    """Cells where an object can be placed: not under / next to an object, and not in the rows (y band) of objects
    that still have to be moved. Falls back to a less conservative map if nothing would be left."""
    original = binary.copy()
    binary = binary.copy()
    rows = np.where((binary == 1) & (on_target != -1))[0]
    binary[rows, :] = 1  # a misplaced object blocks its whole row (the blade comes in along x)
    binary = np.maximum(binary, ndimage.shift(binary, [-2, 0], order=0))
    binary = np.maximum(binary, ndimage.shift(binary, [2, 0], order=0))
    for _ in range(4):
        binary = np.maximum(binary, np.pad(binary, ((0, 0), (1, 0)), mode='constant')[:, :-1])
    for _ in range(4):
        binary = np.maximum(binary, np.pad(binary, ((0, 0), (0, 1)), mode='constant')[:, 1:])
    if np.all(binary == 1):
        binary = original

    def carve(binary, side, top):
        free = np.clip(np.ones_like(binary) - binary, 0, None)
        free[:, 0:8] = 0
        free[:, -side:] = 0
        free[0:top, :] = 0
        free[-top:, :] = 0
        return free

    free = carve(binary, 8, 5)
    if np.all(free == 0):
        free = carve(original, 6, 4)
    return free


def pad_maps(colormap, heightmap):
    """Add the 4 px border (rows both sides, columns left): black colour, 0.3 m (wall height) in the heightmap."""
    padding = ((PAD, PAD), (PAD, 0))
    channels = [np.pad(colormap[:, :, c], padding, 'constant', constant_values=0) for c in range(3)]
    return np.stack(channels, axis=-1), np.pad(heightmap, padding, 'constant', constant_values=0.3)


def build_observation(render, object_names, ws, target_rows, split_instances=False):
    """All observation maps from one camera render.

    Returns a dict with 'colormap' / 'depthmap' (padded) and 'idmap' / 'classmap' / 'freespace' (not padded), and
    'instance_origin' ({id: original id} of split instances) when `split_instances` is set.
    """
    color, depth, segm, position, rotation, intrinsics = render
    color, objects = recolor_by_class(color, segm, object_names)
    points = camera_to_world(depth_to_points(depth, intrinsics), position, rotation)
    heightmap, colormap = top_down_maps(points, color, ws, target_rows)
    id_map, class_map, binary, on_target, origin = object_maps(points, objects, object_names, ws, target_rows,
                                                               split_instances)
    colormap, heightmap = pad_maps(colormap, heightmap)
    obs = {'colormap': colormap, 'depthmap': heightmap, 'idmap': id_map, 'classmap': class_map,
           'freespace': free_space_map(binary, on_target)}
    if split_instances:
        obs['instance_origin'] = origin
    return obs


def relabel_colormap(colormap, id_map, ids, classes):
    """Copy of a colormap in which the objects `ids` are recoloured with the colour of the class `classes`
    (hindsight relabelling). colormap is padded by PAD px while id_map is not: the un-padded window is recoloured."""
    by_label = {LABELS[name]: class_rgb255(name) for name in COLORS}
    image = colormap.copy()
    rows, cols = id_map.shape[:2]
    window = image if image.shape[:2] == (rows, cols) else image[PAD:PAD + rows, PAD:PAD + cols]
    for object_id, label in zip(ids, classes):
        if label in by_label:
            window[id_map == object_id] = by_label[label]
    return image
