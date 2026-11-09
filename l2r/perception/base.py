"""Perception interface.

The policy sees colour-coded class maps built from object masks. In simulation the masks can be read from PyBullet
(ground truth); a real system needs a segmentation module (the paper assumes one). Both implement `segment`.

What the module has to deliver per camera image:
  * instances: HxW int, a unique id per visible object, 0 for background / robot / cabinet,
  * names: {instance id: 'class Type'}  (only the class is used downstream: it selects the colour / target region).
The class of an object is part of the *task* (which region it belongs to), not of its appearance: a network predicts the
object type and the episode's specification (type -> class) turns it into a class.
`to_object` (instance id -> PyBullet object id) is privileged information, used only for the learning signals that need it
(hindsight relabelling) and for diagnostics, never by the policy.
"""
from dataclasses import dataclass


@dataclass
class Segmentation:
    instances: object  # HxW int array
    names: dict  # instance id -> 'class Type'
    to_object: dict = None  # instance id -> PyBullet object id (privileged); None if ids already are object ids


class GroundTruthPerception:
    """Object ids straight from the simulator's segmentation buffer."""

    is_ground_truth = True

    def segment(self, color, depth, object_ids, scene):
        return Segmentation(object_ids, dict(scene.names), None)


def make_perception(params):
    """Perception selected by `SimParams.perception` ('gt' | 'network')."""
    if params.perception == 'network':
        from l2r.perception.network import NetworkPerception
        return NetworkPerception(params.perception_weights)
    return GroundTruthPerception()
