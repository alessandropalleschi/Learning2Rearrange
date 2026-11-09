"""Object-agnostic perception: the network knows nothing about object types.

The U-Net predicts three classes per pixel: background, object interior, and object *boundary* (pixels where two objects touch).
Instances are the connected components of the interior, grown back over the boundary pixels. No type catalogue is involved, so
the masks carry over to objects never seen in training (`train_perception.py ... --task agnostic`, evaluated with held-out types).

The class of an object is a property of the task, not of its pixels. It is assigned by matching every predicted instance to the
*reference descriptors* of the objects in the episode specification (mean colour and footprint area of each object as shown at
the start of the episode, the 'reference image' of the object to sort): an optimal one-to-one assignment (Hungarian). In the
simulator the references are rendered from the first frame of the episode, so class assignment is an upper bound of what a
reference-image matcher achieves; the segmentation itself is learned.
"""
import numpy as np
import torch
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

from l2r.perception.base import Segmentation
from l2r.perception.network import prepare_input, INPUT_SCALE
from l2r.perception.unet import SegmentationNet

BACKGROUND, INTERIOR, BOUNDARY = 0, 1, 2
MAX_COST = 0.6  # descriptor distance above which a predicted instance is not assigned to any object (it is dropped)


def agnostic_targets(ids, names):
    """Ground truth for the network from an id image: 0 background, 1 interior, 2 pixels of an object next to another object."""
    objects = np.isin(ids, list(names))
    target = objects.astype(np.uint8)
    other = np.zeros(ids.shape, bool)
    for dy, dx in ((0, 1), (1, 0)):
        a, b = ids[:ids.shape[0] - dy, :ids.shape[1] - dx], ids[dy:, dx:]
        touching = objects[:ids.shape[0] - dy, :ids.shape[1] - dx] & objects[dy:, dx:] & (a != b)
        other[:ids.shape[0] - dy, :ids.shape[1] - dx] |= touching
        other[dy:, dx:] |= touching
    target[other] = BOUNDARY
    return target


def descriptor(color, region):
    """(mean RGB in [0, 1], footprint area in px) of a mask."""
    return np.append(color[region].mean(0) / 255.0, 0.0), float(region.sum())


def descriptor_cost(a, b):
    """Distance between two descriptors: colour (L2) plus the log ratio of the areas (objects are partly occluded: weak term)."""
    return float(np.linalg.norm(a[0][:3] - b[0][:3]) + 0.1 * abs(np.log(max(a[1], 1.0) / max(b[1], 1.0))))


class AgnosticPerception:
    is_ground_truth = False

    def __init__(self, weights=None, model=None, min_area=40, device='cpu'):
        self.min_area, self.device = min_area, device
        if model is None:
            model = SegmentationNet(2)
            model.load_state_dict(torch.load(weights, map_location=device))
        self.model = model.to(device).eval()
        self._references, self._key = {}, None

    @torch.no_grad()
    def predict(self, color, depth):
        """HxW uint8 in {BACKGROUND, INTERIOR, BOUNDARY}."""
        x = prepare_input(color, depth)[None].to(self.device)
        out = self.model(x).argmax(1)[0].cpu().numpy().astype(np.uint8)
        return np.repeat(np.repeat(out, INPUT_SCALE, axis=0), INPUT_SCALE, axis=1)[:color.shape[0], :color.shape[1]]

    @staticmethod
    def instances_from_prediction(pred, min_area=40):
        """Class-agnostic instance ids (0 = background) from the three-class prediction."""
        labels, count = ndimage.label(pred == INTERIOR)
        areas = ndimage.sum(np.ones_like(labels), labels, index=np.arange(1, count + 1))
        for k, area in enumerate(areas, 1):
            if area < min_area:
                labels[labels == k] = 0
        objects = pred > 0
        if not (labels > 0).any():
            return np.zeros(pred.shape, np.int64)
        _, (iy, ix) = ndimage.distance_transform_edt(labels == 0, return_indices=True)
        grown = labels[iy, ix]  # the nearest interior label, kept on object pixels only
        return np.where(objects, grown, 0).astype(np.int64)

    def _reference(self, color, object_ids, scene):
        key = tuple(sorted(scene.names.items()))
        if key != self._key:  # a new episode: describe its objects as they look now (the specification)
            self._key = key
            self._references = {uid: descriptor(color, object_ids == uid) for uid in scene.names if (object_ids == uid).any()}
        return self._references

    def segment(self, color, depth, object_ids, scene):
        pred = self.predict(color, depth)
        instances = self.instances_from_prediction(pred, self.min_area)
        references = self._reference(color, object_ids, scene)
        found = [int(i) for i in np.unique(instances) if i != 0]
        uids = list(references)
        out, names, to_object, next_id = np.zeros(instances.shape, np.int64), {}, {}, 1
        if found and uids:
            cost = np.array([[descriptor_cost(descriptor(color, instances == i), references[u]) for u in uids] for i in found])
            for r, c in zip(*linear_sum_assignment(cost)):
                if cost[r, c] > MAX_COST:
                    continue
                region = instances == found[r]
                out[region] = next_id
                names[next_id] = scene.names[uids[c]]
                overlap = [(int((object_ids[region] == uid).sum()), uid) for uid in scene.names]  # privileged: diagnostics only
                to_object[next_id] = max(overlap)[1] if overlap and max(overlap)[0] > 0 else None
                next_id += 1
        return Segmentation(out, names, to_object)
