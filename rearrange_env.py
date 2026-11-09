"""Gym-style wrapper around the PyBullet simulation (`env.Environment`).

    env = RearrangeEnv(cfg, gui=False, num_obj=1)
    obs = env.reset()
    obs, reward, terminated, truncated, info = env.step({'primitive': 'push', 'pix': (rot, y, x)})

It owns everything that is not learning: action execution, feasibility checks, the reward (paper Eq. 2),
episode termination and the hindsight relabelling of observations.
"""
import numpy as np
from skimage.draw import line

from config import Config
from env import Environment


def check_push(env, depth, pix_x, pix_y, angle, push_length=0.08):
    """True if there is something in the way of the push (there must be an object to push)."""
    primitive_position_f = env.px_to_xy(pix_x, pix_y)
    primitive_position_f[1] = primitive_position_f[1] + push_length * np.sin(angle)
    depth_new = depth.copy()
    depth_new = depth_new[4:depth_new.shape[0] - 4, 4:depth_new.shape[1]]
    pixel_final = env.xyz_to_pix(primitive_position_f)
    end_px = min(pixel_final[0], depth_new.shape[0] - 1)
    rr, cc = line(pix_y, pix_x, end_px, pix_x)
    local_region_push = depth_new[rr, pix_x:]
    local_region_push = local_region_push[local_region_push > 0.02]
    return len(local_region_push) > 2


def select_placement(y_up, y_down, freespace, no_obj=False):
    """Placement heuristic: free pixel close to the middle of the target region and to the back of the cabinet."""
    freespace = freespace.copy()
    freespace[:, 0:5] = 0
    freespace[:, -5:] = 0
    ind = np.where(freespace > 0)
    if ind[0].size == 0:  # no free pixel left
        return freespace.shape[1] // 2, (y_up + y_down) // 2
    if no_obj:
        dist_y = np.abs(ind[0] - freespace.shape[0] // 2) + 1 / np.maximum(ind[1], 1)
    else:
        dist_y = np.abs(ind[0] - (y_up + y_down) // 2) + ind[1]
    k = np.nanargmin(dist_y)
    return ind[1][k], ind[0][k]


def get_place_pose(env, obj_class, freespace):
    """Target region index follows the order of the environment's target dict (blue=1, yellow=5, green=3)."""
    y_down, y_up, _ = env.add_target_pixels()
    index = {1: 0, 5: 1, 3: 2}[int(obj_class)]
    return select_placement(y_up[index], y_down[index], freespace)


class RearrangeEnv:
    def __init__(self, cfg=None, gui=False, num_obj=1):
        self.cfg = cfg or Config()
        self.num_obj = num_obj
        self.sim = Environment(gui=gui, num_obj=[num_obj])
        self.obs = None
        self.actions = 0
        self.episode_return = 0.0

    # ------------------------------------------------------------------ gym API
    def reset(self, num_obj=None):
        if num_obj is not None:
            self.num_obj = num_obj
            self.sim.num_obj = [num_obj]
        while True:  # resample until the scene is valid and not already sorted
            if self.sim.reset() and self.sim.check_sorted() < 1:
                break
        self.actions = 0
        self.episode_return = 0.0
        self.obs = self.sim.get_observation()
        self._relabel_state = self._relabel_status()
        return self.obs

    def step(self, action):
        """action: {'primitive': 'push'|'grasp', 'pix': (rotation_index, row, col)} in Q-map coordinates."""
        obs, sim, cfg = self.obs, self.sim, self.cfg
        prim, pix = action['primitive'], action['pix']
        px_x, px_y = pix[2], pix[1]
        rotation = np.deg2rad(pix[0] * (360.0 / 2) - 90)  # two push directions: -90 / +90 degrees
        position = sim.px_to_xy(px_x, px_y)
        executed, place_pix = False, None

        if prim == 'push' and check_push(sim, obs['depthmap'], px_x, px_y, rotation, cfg.push_length):
            executed = sim.push(position, rotation)
        elif prim == 'grasp' and obs['classmap'][px_y, px_x] != 0:
            executed = sim.pick(position)
            if executed:
                obj_class = obs['classmap'][px_y, px_x]
                freespace = sim.get_observation()['freespace']
                place_pix = get_place_pose(sim, obj_class, freespace)
                executed = sim.place(sim.px_to_xy(*place_pix))

        sim.wait_static()
        sorted_ = sim.check_sorted() == 1
        valid = sim.check_sim()
        self.actions += 1

        if not valid:
            reward = cfg.reward_invalid
        elif sorted_:
            reward = cfg.reward_goal
        else:
            reward = cfg.reward_step
        terminated = sorted_ or not valid
        truncated = (not terminated) and self.actions >= cfg.max_actions
        self.episode_return += reward

        next_obs = sim.get_observation() if not terminated else obs
        prev_relabel = self._relabel_state
        self._relabel_state = self._relabel_status()
        # Hindsight relabelling (paper Fig. 8): all objects are inside *some* target region but not (all) in
        # their own one; reassigning the classes would make this a sorted state.
        relabel = (reward == cfg.reward_step and self._relabel_state[0]
                   and (not prev_relabel[0] or prev_relabel[1] != self._relabel_state[1]))
        info = {'executed': bool(executed), 'sorted': bool(sorted_ and valid), 'invalid': not valid,
                'place_pix': place_pix, 'relabel': bool(relabel), 'episode_return': self.episode_return,
                'actions': self.actions}
        self.obs = next_obs
        return next_obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------ hindsight relabelling
    def _relabel_status(self):
        possible = self.sim.check_relabel() == 1
        return possible, list(self.sim.new_class) if possible else None

    def relabel(self, obs, next_obs):
        """Copies of (obs, next_obs) whose colormaps use the post-hoc class assignment of the current scene."""
        sim = self.sim
        new_obs, new_next = dict(obs), dict(next_obs)
        new_obs['colormap'] = sim.relabel_image([obs['colormap']], [obs['idmap']], sim.id_list, sim.new_class)[0]
        new_next['colormap'] = sim.relabel_image([next_obs['colormap']], [next_obs['idmap']], sim.id_list, sim.new_class)[0]
        return new_obs, new_next
