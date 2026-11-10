"""The RGB-D camera of `l2r.sim.camera` rendered by MuJoCo (same pose, intrinsics, noise and outputs)."""
import numpy as np

from l2r.sim.camera import Camera


class MujocoCamera(Camera):
    def render(self):
        p, cfg = self.p, self.config
        height, width = cfg['image_size']
        rotm = np.float32(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(cfg['rotation']))).reshape(3, 3)
        look = rotm @ np.float32([0, 0, 1])
        up = rotm @ np.float32([0, -1, 0])
        # MuJoCo cameras look along their -z with +y up in the image
        z = -look / np.linalg.norm(look)
        y = up - (up @ look) * look
        y = y / np.linalg.norm(y)
        x = np.cross(y, z)
        vertical_fov = float(np.degrees(2 * np.arctan((height / 2) / cfg['intrinsics'][0])))
        color, depth, segm = p.render_camera(np.asarray(cfg['position'], float), np.column_stack([x, y, z]), vertical_fov,
                                             height, width)
        if cfg['noise']:
            color = np.uint8(np.clip(np.int32(color) + np.int32(np.random.normal(0, 3, color.shape)), 0, 255))
            depth = depth + np.random.normal(0, self.params.depth_noise_std, (height, width))
        return color, depth, segm, cfg['position'], rotm, np.float32(cfg['intrinsics']).reshape(3, 3)
