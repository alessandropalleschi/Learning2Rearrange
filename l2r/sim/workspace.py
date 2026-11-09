"""Geometry of the cabinet workspace and its top-down heightmap."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Workspace:
    """Axis-aligned workspace in front of the robot (metres) and the pixel grid of the heightmaps.

    The heightmap has one row per `pixel_size` along y (cabinet width) and one column per `pixel_size` along x
    (cabinet depth), so pixel (row, col) = (y index, x index).
    """

    depth: float = 0.5  # x extent
    width: float = 0.6  # y extent
    height: float = 0.35  # z extent
    floor_z: float = 0.4  # height of the cabinet floor above the ground
    pixel_size: float = 0.005

    @classmethod
    def from_params(cls, params):
        return cls(depth=params.cabinet_depth, height=params.cabinet_height)

    @property
    def bounds(self):
        """3x2 float32 array [[x_min, x_max], [y_min, y_max], [z_min, z_max]] (float32 like the original code)."""
        return np.float32([[-self.depth / 2, self.depth / 2], [-self.width / 2, self.width / 2],
                           [self.floor_z, self.floor_z + self.height]])

    @property
    def scene_size(self):
        """(x, y) size of the cabinet floor, slightly larger than the workspace."""
        b = self.bounds
        return [b[0][1] - b[0][0] + 0.005, b[1][1] - b[1][0] + 0.015]

    @property
    def heightmap_shape(self):
        """(rows, cols) of the un-padded heightmap."""
        b = self.bounds
        return (int(np.round((b[1, 1] - b[1, 0]) / self.pixel_size)),
                int(np.round((b[0, 1] - b[0, 0]) / self.pixel_size)))

    def px_to_xy(self, px, py):
        """Heightmap pixel (col px, row py) -> workspace (x, y) in metres."""
        b = self.bounds
        return [px * self.pixel_size + b[0][0], py * self.pixel_size + b[1][0]]

    def y_to_row(self, y):
        """Workspace y (m) -> heightmap row."""
        return int(np.round((y - self.bounds[1, 0]) / self.pixel_size))
