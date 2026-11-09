"""Segmentation-network perception: object types per pixel -> classes (task specification) -> instances."""
import numpy as np
import torch
from scipy import ndimage

from l2r.perception.base import Segmentation
from l2r.perception.unet import SegmentationNet
from l2r.sim.constants import OBJECT_CATALOGUE

INPUT_SCALE = 2  # the network works on the image downsampled by this factor, its output is upsampled (nearest)
DEPTH_NORM = 1.5  # m


def prepare_input(color, depth, scale=INPUT_SCALE):
    """RGB uint8 + metric depth -> (4, H/scale, W/scale) float32 tensor."""
    x = np.concatenate([color[::scale, ::scale].astype(np.float32) / 255.0,
                        (depth[::scale, ::scale].astype(np.float32) / DEPTH_NORM)[..., None]], axis=-1)
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


class NetworkPerception:
    """Masks from a `SegmentationNet`. The object *types* it predicts are mapped to classes by the episode's
    specification (which type belongs to which class); a type that is not part of the episode is ignored. Instances are
    the connected components of every type (two touching objects of one type merge: a known limitation)."""

    is_ground_truth = False

    def __init__(self, weights=None, model=None, min_area=40, device='cpu'):
        self.min_area, self.device = min_area, device
        if model is None:
            model = SegmentationNet(len(OBJECT_CATALOGUE))
            model.load_state_dict(torch.load(weights, map_location=device))
        self.model = model.to(device).eval()

    @torch.no_grad()
    def predict_types(self, color, depth):
        """HxW uint8: 0 = background, i = OBJECT_CATALOGUE[i - 1]."""
        x = prepare_input(color, depth)[None].to(self.device)
        types = self.model(x).argmax(1)[0].cpu().numpy().astype(np.uint8)
        return np.repeat(np.repeat(types, INPUT_SCALE, axis=0), INPUT_SCALE, axis=1)[:color.shape[0], :color.shape[1]]

    def segment(self, color, depth, object_ids, scene):
        return self.instances_from_types(self.predict_types(color, depth), object_ids, scene)

    def instances_from_types(self, types, object_ids, scene):
        type_to_class = {name.split(' ')[1]: name.split(' ')[0] for name in scene.names.values()}
        instances, names, to_object, next_id = np.zeros(types.shape, np.int64), {}, {}, 1
        for index, type_name in enumerate(OBJECT_CATALOGUE, 1):
            if type_name not in type_to_class:
                continue
            labels, count = ndimage.label(types == index)
            for k in range(1, count + 1):
                region = labels == k
                if region.sum() < self.min_area:
                    continue
                instances[region] = next_id
                names[next_id] = f'{type_to_class[type_name]} {type_name}'
                overlap = [(int((object_ids[region] == uid).sum()), uid) for uid in scene.names]  # privileged match
                to_object[next_id] = max(overlap)[1] if overlap and max(overlap)[0] > 0 else None
                next_id += 1
        return Segmentation(instances, names, to_object)
