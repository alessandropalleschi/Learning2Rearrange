"""Object classes, colours and target regions of the rearrangement task."""
import numpy as np

CLASSES = ('blue', 'green', 'yellow')  # the three classes of the task
TARGET_ORDER = ('blue', 'yellow', 'green')  # order in which the shuffled target regions are assigned to the classes

COLORS = {  # RGBA of the class colours
    'blue': (78 / 255, 121 / 255, 167 / 255, 1.0),
    'green': (89 / 255, 169 / 255, 79 / 255, 1.0),
    'yellow': (237 / 255, 201 / 255, 72 / 255, 1.0),
}
LABELS = {'blue': 1, 'green': 3, 'yellow': 5}  # class label in the label / class maps (0 = background)
TARGET_COLORS = {'blue': [0, 0, 139], 'green': [0, 110, 51], 'yellow': [246, 190, 0]}  # recolouring of the target regions
OCCLUDED_RGB = 105  # grey of cells the camera cannot see

# YCB objects the scenes are drawn from (three distinct ones per episode, one per class)
OBJECT_CATALOGUE = ['PottedMeatCan', 'CrackerBox', 'ChipsCan', 'MustardBottle']
MAX_OBJECTS_PER_CLASS = 5
BLOCK_HALF_EXTENTS = {'blue': [0.025, 0.025, 0.05], 'green': [0.025, 0.025, 0.05], 'yellow': [0.025, 0.025, 0.05]}


def class_rgb255(name):
    """Class colour as float RGB in 0..255."""
    return np.multiply(COLORS[name][:3], 255)


def goal_slot_choices(bounds, grid=(2, 3), margin=0.01):
    """Rectangular goal slots for `--goal_layout grid`: the cabinet floor split into grid[0] (depth, x) x grid[1] (width, y) cells;
    returns [[x_min, x_max, y_min, y_max], ...] (m). Three of them are assigned to the classes per episode."""
    (x0, x1), (y0, y1) = bounds[0], bounds[1]
    xs, ys = np.linspace(x0, x1, grid[0] + 1), np.linspace(y0, y1, grid[1] + 1)
    return [[xs[i] + margin, xs[i + 1] - margin, ys[j] + margin, ys[j + 1] - margin] for i in range(grid[0]) for j in range(grid[1])]


def target_region_choices(bounds):
    """Candidate y-intervals (m) of the three target regions; they are randomly assigned to the classes."""
    y_min, y_max = bounds[1][0], bounds[1][1]
    return [[y_min + 0.01, y_min + 0.16], [y_max - 0.18, y_max - 0.01], [-0.075, 0.085]]
