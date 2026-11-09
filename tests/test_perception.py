"""Perception: type map -> classes (task specification) -> instances, the privileged match, the training metrics."""
from types import SimpleNamespace

import numpy as np
import torch

from l2r.perception.network import INPUT_SCALE, NetworkPerception
from l2r.perception.training import augment, iou_from_confusion, confusion, match_instances
from l2r.perception.unet import SegmentationNet
from l2r.sim.constants import OBJECT_CATALOGUE


def test_instances_from_types_use_the_episode_specification():
    perception = NetworkPerception(model=SegmentationNet(len(OBJECT_CATALOGUE)), min_area=10)
    scene = SimpleNamespace(names={7: 'blue CrackerBox', 9: 'green ChipsCan'})
    cracker, chips, meat = (OBJECT_CATALOGUE.index(n) + 1 for n in ('CrackerBox', 'ChipsCan', 'PottedMeatCan'))
    types, truth = np.zeros((40, 60), np.uint8), np.zeros((40, 60), np.int64)
    types[5:15, 5:15], truth[5:15, 5:15] = cracker, 7  # a blue object
    types[20:30, 5:15], truth[20:30, 5:15] = chips, 9  # a green object
    types[5:15, 30:40] = cracker  # a second blue instance (disjoint component, no ground truth behind it)
    types[20:30, 40:50] = meat  # a type that is not part of this episode: ignored
    types[35:37, 55:57] = chips  # below min_area: dropped
    seg = perception.instances_from_types(types, truth, scene)
    classes = sorted(name.split(' ')[0] for name in seg.names.values())
    assert classes == ['blue', 'blue', 'green'] and len(set(seg.instances[seg.instances > 0])) == 3
    assert (seg.instances[20:30, 40:50] == 0).all() and (seg.instances[35:37, 55:57] == 0).all()
    by_object = {obj: inst for inst, obj in seg.to_object.items() if obj is not None}
    assert set(by_object) == {7, 9} and seg.names[by_object[7]].startswith('blue')  # privileged match to the true objects


def test_network_runs_on_the_camera_image_size_and_returns_full_resolution_types():
    model = SegmentationNet(len(OBJECT_CATALOGUE))
    perception = NetworkPerception(model=model)
    color, depth = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8), np.random.uniform(0.3, 1.2, (480, 640))
    types = perception.predict_types(color, depth)
    assert types.shape == (480, 640) and types.max() <= len(OBJECT_CATALOGUE) and INPUT_SCALE == 2


def test_instance_matching_and_iou_metrics():
    gt = np.zeros((30, 30), np.int64)
    gt[2:12, 2:12], gt[15:25, 15:25] = 5, 6
    pred = np.zeros_like(gt)
    pred[2:12, 2:12] = 1  # exact
    pred[15:25, 15:20] = 2  # half of the second object: IoU 0.5 -> still a match
    pred[27:30, 27:30] = 3  # a false positive
    found, total, matched, predicted = match_instances(pred, gt, min_area=10)
    assert (found, total, matched, predicted) == (2, 2, 2, 3)
    c = confusion(torch.tensor([[0, 1, 1, 2]]), torch.tensor([[0, 1, 2, 2]]), 3)
    assert torch.allclose(iou_from_confusion(c), torch.tensor([1.0, 0.5, 0.5]))


def test_augmentation_keeps_labels_aligned_with_the_image():
    generator = torch.Generator().manual_seed(0)
    x, y = torch.rand(6, 4, 8, 8), torch.randint(0, 3, (6, 8, 8))
    x2, y2 = augment(x, y.clone(), generator)
    flipped = [i for i in range(6) if not torch.equal(y2[i], y[i])]
    assert flipped and all(torch.equal(y2[i], y[i].flip(-1)) for i in flipped)
