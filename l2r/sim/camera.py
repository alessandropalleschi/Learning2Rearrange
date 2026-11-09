"""Simulated RGB-D camera in front of the cabinet."""
import numpy as np

IMAGE_SIZE = (480, 640)  # (height, width)
DEFAULT_CONFIG = {
    'rotation': (np.pi / 3, np.pi, -np.pi / 2),
    'intrinsics': (450., 0, IMAGE_SIZE[1] / 2, 0, 450., IMAGE_SIZE[0] / 2, 0, 0, 1),
    'image_size': IMAGE_SIZE,
    'zrange': (0.01, 10.0),  # near / far plane (m)
    'noise': False,
}


class Camera:
    """Renders colour, metric depth and the object-id image; poses and intrinsics follow `SimParams`."""

    def __init__(self, p, params, workspace, gui=False):
        self.p, self.params, self.ws, self.gui = p, params, workspace, gui
        self.config = self._make_config()

    def randomize(self):
        """New camera pose / noise setting for the next episode (domain randomisation, see SimParams)."""
        self.config = self._make_config()

    def _make_config(self):
        params, cfg = self.params, dict(DEFAULT_CONFIG)
        cfg['position'] = (0.52, 0, self.ws.bounds[2][0] + 0.38)  # in front of the cabinet, 0.38 m above its floor
        if params.randomize:
            cfg['position'] = tuple(np.asarray(cfg['position']) +
                                    np.random.uniform(-params.cam_pos_jitter, params.cam_pos_jitter, 3))
            cfg['rotation'] = tuple(np.asarray(DEFAULT_CONFIG['rotation']) +
                                    np.deg2rad(np.random.uniform(-params.cam_rot_jitter_deg, params.cam_rot_jitter_deg, 3)))
        cfg['noise'] = bool(params.sensor_noise)
        s = params.camera_scale
        if s != 1.0:  # fewer pixels: scale the image and the intrinsics together
            cfg['image_size'] = (int(IMAGE_SIZE[0] * s), int(IMAGE_SIZE[1] * s))
            fx, _, cx, _, fy, cy = DEFAULT_CONFIG['intrinsics'][:6]
            cfg['intrinsics'] = (fx * s, 0, cx * s, 0, fy * s, cy * s, 0, 0, 1)
        return cfg

    def render(self):
        """Returns color (HxWx3 uint8), depth (HxW, metres), segm (HxW int object uid), the camera position, its
        3x3 rotation matrix (camera -> world) and the 3x3 intrinsics matrix."""
        p, cfg = self.p, self.config
        height, width = cfg['image_size']
        quaternion = p.getQuaternionFromEuler(cfg['rotation'])
        rotm = np.float32(p.getMatrixFromQuaternion(quaternion)).reshape(3, 3)
        lookdir = (rotm @ np.float32([0, 0, 1]).reshape(3, 1)).reshape(-1)
        updir = (rotm @ np.float32([0, -1, 0]).reshape(3, 1)).reshape(-1)
        lookat = cfg['position'] + lookdir
        znear, zfar = cfg['zrange']
        view = p.computeViewMatrix(cfg['position'], lookat, updir)
        vertical_fov = (np.arctan((height / 2) / cfg['intrinsics'][0]) * 2 / np.pi) * 180  # degrees
        projection = p.computeProjectionMatrixFOV(vertical_fov, width / height, znear, zfar)
        _, _, color, zbuffer, segm = p.getCameraImage(
            width=width, height=height, viewMatrix=view, projectionMatrix=projection,
            shadow=0,  # the RGB image is recoloured by class afterwards: shadows only cost render time
            flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            # OpenGL needs a display; the CPU renderer works headless (containers, CI)
            renderer=p.ER_BULLET_HARDWARE_OPENGL if self.gui else p.ER_TINY_RENDERER)

        color = np.array(color, dtype=np.uint8).reshape((height, width, 4))[:, :, :3]  # drop alpha
        if cfg['noise']:
            color = np.uint8(np.clip(np.int32(color) + np.int32(np.random.normal(0, 3, color.shape)), 0, 255))
        # OpenGL z-buffer in [0, 1] -> metric depth
        z = zfar + znear - (2.0 * np.array(zbuffer).reshape((height, width)) - 1.0) * (zfar - znear)
        depth = (2.0 * znear * zfar) / z
        if cfg['noise']:
            depth += np.random.normal(0, self.params.depth_noise_std, (height, width))
        # object uid in the low 24 bits, link index + 1 above them (background = -1): keep the object uid only
        segm = np.asarray(segm, dtype=np.int64).reshape((height, width)) & 0xFFFFFF
        return color, depth, segm, cfg['position'], rotm, np.float32(cfg['intrinsics']).reshape(3, 3)
