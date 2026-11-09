"""From the camera image to the maps the policy sees: point cloud -> top-down heightmap / colormap, per-object
class and id maps, and the free-space map used by the placement heuristic. Pure numpy / OpenCV, no simulator."""
import cv2
import numpy as np
from scipy import ndimage

from l2r.sim.constants import COLORS, LABELS, OCCLUDED_RGB, TARGET_ORDER, class_rgb255

PAD = 4  # the colormap / heightmap returned to the policy get a 4 px border (rows: both sides, columns: left)


def depth_to_points(depth, intrinsics):
    """HxW metric depth -> HxWx3 points in camera coordinates."""
    height, width = depth.shape
    px, py = np.meshgrid(np.linspace(0, width - 1, width), np.linspace(0, height - 1, height))
    px = (px - intrinsics[0, 2]) * (depth / intrinsics[0, 0])
    py = (py - intrinsics[1, 2]) * (depth / intrinsics[1, 1])
    return np.float32([px, py, depth]).transpose(1, 2, 0)


def transform_points(points, transform):
    """Apply a 4x4 rigid transform to HxWx3 points (in place, like the original code) and return them."""
    homogeneous = np.pad(points.copy(), ((0, 0), (0, 0), (0, 1)), 'constant', constant_values=1)
    for i in range(3):
        points[..., i] = np.sum(transform[i, :] * homogeneous, axis=-1)
    return points


def camera_to_world(points, position, orientation_matrix):
    transform = np.eye(4)
    transform[:3, :] = np.hstack((orientation_matrix, np.float32(position).reshape(3, 1)))
    return transform_points(points, transform)


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


def mask_colormap(points, mask_image, ws):
    """Colour (class colour of one object) visible at the top of every heightmap cell for a per-object mask image."""
    bounds = ws.bounds
    points, mask_image, px, py, (height, width) = _project(points, mask_image, bounds, ws.pixel_size, inclusive=False)
    colormap = np.zeros((height, width, mask_image.shape[-1]), dtype=np.uint8)
    for c in range(mask_image.shape[-1]):
        colormap[py, px, c] = mask_image[:, c]
    return colormap


def class_masks(color, segm, object_names):
    """Recolour the camera image by object class and build one mask image per object.

    Args:
        color: HxWx3 uint8 image (modified in place). segm: HxW int object uid of every pixel.
        object_names: {object uid: 'class YcbName'}.
    Returns:
        color, masks (NxHxWx3 uint8 class-coloured image per object, None without objects), class labels, colours
        and object ids of the masks. The label of every pixel is not needed outside, so no label map is built; ids
        are never rewritten in place.
    """
    known = np.isin(segm, list(object_names))
    color[~known] = 255
    masks, labels, colors, ids = [], [], [], []
    for uid in np.unique(segm[known]):
        name = object_names[uid].split(' ')[0]
        rgb = class_rgb255(name)
        region = segm == uid
        color[region] = rgb
        image = np.zeros_like(color)
        image[region] = rgb
        masks.append(image)
        labels.append(LABELS[name])
        colors.append(rgb)
        ids.append(uid)
    return color, (np.stack(masks) if masks else None), np.array(labels), np.array(colors), np.array(ids)


def object_maps(points, masks, labels, colors, ids, ws, target_rows):
    """Per-object maps on the heightmap grid: object id, class label, and which cells belong to objects that
    already stand inside their own target region."""
    shape = ws.heightmap_shape
    id_map, class_map = np.zeros(shape, np.float32), np.zeros(shape, np.float32)
    binary, on_target = np.zeros(shape, np.float32), np.zeros(shape, np.float32)
    if masks is None:
        return id_map, class_map, binary, on_target
    low, high = target_rows[0], target_rows[1]
    for mask_image, label, rgb, uid in zip(masks, labels, colors, ids):
        visible = mask_colormap(points, mask_image, ws)
        cells = (visible[:, :, 0] == rgb[0]) & (visible[:, :, 1] == rgb[1]) & (visible[:, :, 2] == rgb[2])
        occupancy = cells.astype(np.uint8)
        moments = cv2.moments(occupancy, binaryImage=True)
        if moments['m00'] == 0:
            continue
        row = int(moments['m01'] / moments['m00'])  # centroid row
        k = TARGET_ORDER.index([name for name in TARGET_ORDER if LABELS[name] == label][0])  # region of its class
        id_map[cells], class_map[cells], binary[cells] = uid, label, 1
        if low[k] <= row <= high[k]:
            on_target[cells] = -1
    return id_map, class_map, binary, on_target


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


def build_observation(render, object_names, ws, target_rows):
    """All observation maps from one camera render.

    Returns a dict with 'colormap' / 'depthmap' (padded) and 'idmap' / 'classmap' / 'freespace' (not padded).
    """
    color, depth, segm, position, rotation, intrinsics = render
    color, masks, labels, colors, ids = class_masks(color, segm, object_names)
    points = camera_to_world(depth_to_points(depth, intrinsics), position, rotation)
    heightmap, colormap = top_down_maps(points, color, ws, target_rows)
    id_map, class_map, binary, on_target = object_maps(points, masks, labels, colors, ids, ws, target_rows)
    colormap, heightmap = pad_maps(colormap, heightmap)
    return {'colormap': colormap, 'depthmap': heightmap, 'idmap': id_map, 'classmap': class_map,
            'freespace': free_space_map(binary, on_target)}


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
