"""Where the object masks and classes of the observation come from: PyBullet ground truth or a segmentation network."""
from l2r.perception.base import GroundTruthPerception, Segmentation, make_perception

__all__ = ['GroundTruthPerception', 'Segmentation', 'make_perception']
