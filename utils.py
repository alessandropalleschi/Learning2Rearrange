import struct
import math
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pdb

# Third-party libraries
import pybullet
import pybullet_data
from pybullet_utils import bullet_client as bc
import numpy as np


def check_env_depth_change(prev_depth_heightmap, depth_heightmap, change_threshold=300):
    depth_diff = abs(prev_depth_heightmap-depth_heightmap)
    depth_diff[np.isnan(depth_diff)] = 0
    depth_diff[depth_diff > 0.2] = 0
    depth_diff[depth_diff < 0.01] = 0
    depth_diff[depth_diff > 0] = 1
    change_value = np.sum(depth_diff)
    change_detected = change_value > change_threshold

    return change_detected, change_value
