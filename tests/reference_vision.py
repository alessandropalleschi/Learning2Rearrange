"""The previous (per-object) implementation of the object maps, kept as the reference the single-pass version
in l2r.sim.vision is tested against."""
import cv2
import numpy as np

from l2r.sim.constants import COLORS, LABELS, TARGET_ORDER, class_rgb255  # noqa: F401
from l2r.sim.vision import _project


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
