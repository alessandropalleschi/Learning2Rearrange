"""Gym-style wrapper around the PyBullet simulation (`env.Environment`).

    env = RearrangeEnv(cfg, gui=False, num_obj=1)
    obs = env.reset()
    obs, reward, terminated, truncated, info = env.step({'primitive': 'push', 'pix': (rot, y, x)})

It owns everything that is not learning: action execution, feasibility checks, the reward (paper Eq. 2),
episode termination and the hindsight relabelling of observations.
"""
import random

import numpy as np
from skimage.draw import line

from belief import BeliefTracker, compact_belief
from config import Config
from env import Environment, BOUNDS, PIXEL_SIZE


def decode_push(index, push_lengths):
    """Push action index (direction * L + length) -> (angle in rad, length in m). Direction 0/1 = -90/+90 degrees."""
    direction, length = divmod(int(index), len(push_lengths))
    return np.deg2rad(direction * 180.0 - 90), push_lengths[length]


def px_to_xy(px, py):
    """Heightmap pixel -> workspace (x, y) in metres (pure geometry, no simulator needed)."""
    return [px * PIXEL_SIZE + BOUNDS[0][0], py * PIXEL_SIZE + BOUNDS[1][0]]


def y_to_pix(y):
    return int(np.round((y - BOUNDS[1, 0]) / PIXEL_SIZE))


def check_push(depth, pix_x, pix_y, angle, push_length=0.08):
    """True if there is something in the way of the push (there must be an object to push)."""
    y_end = px_to_xy(pix_x, pix_y)[1] + push_length * np.sin(angle)
    depth_new = depth[4:depth.shape[0] - 4, 4:depth.shape[1]]
    end_px = min(y_to_pix(y_end), depth_new.shape[0] - 1)
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
    def __init__(self, cfg=None, gui=False, num_obj=1, params=None):
        self.cfg = cfg or Config()
        self.num_obj = num_obj
        self.sim = Environment(gui=gui, num_obj=[num_obj], params=params)
        self.obs = None  # full (internal) observation of the current state
        self.tracker = BeliefTracker(self.cfg.belief_age_horizon)
        self.actions = 0
        self.episode_return = 0.0
        self.holding = False

    # ------------------------------------------------------------------ gym API
    def reset(self, num_obj=None, seed=None):
        if seed is not None:  # reproducible scene (object sizes, positions, target regions)
            np.random.seed(seed)
            random.seed(seed)
        if num_obj is not None:
            self.num_obj = num_obj
            self.sim.num_obj = [num_obj]
        while True:  # resample until the scene is valid and not already sorted
            if self.sim.reset() and self.sim.check_sorted() < 1:
                break
        self.actions = 0
        self.episode_return = 0.0
        self.holding = False
        self.tracker.reset()
        self.obs = self._full_obs(None)
        self._relabel_state = self._relabel_status()
        return self.compact(self.obs)

    def step(self, action):
        """action: {'primitive': 'push'|'grasp'|'place', 'pix': (rotation_index, row, col)} in Q-map coordinates.

        placement='heuristic': 'grasp' picks and places in one action (placement by rule).
        placement='learned':   'grasp' only picks. While the object is held, the observation (taken after the
                               pick, with the held class and the free cells) is returned with reward 0 and the
                               next action must be 'place'; the reward is given for the complete pick-and-place.
        """
        obs, sim, cfg = self.obs, self.sim, self.cfg
        prim, pix = action['primitive'], action['pix']
        px_x, px_y = pix[2], pix[1]
        position = sim.px_to_xy(px_x, px_y)
        executed, place_pix = False, None
        stops_before = sim.force_stops
        learned = cfg.placement == 'learned'
        if self.holding and prim != 'place':
            raise ValueError("an object is held: the next action must be 'place'")

        if prim == 'push':
            angle, length = decode_push(pix[0], cfg.push_lengths)
            if check_push(obs['depthmap'], px_x, px_y, angle, length):
                executed = sim.push(position, angle, push_length=length)
        elif prim == 'place':
            place_pix = (px_x, px_y)  # (col, row) of the free cell
            executed = sim.place(position)
            self.holding = False
        elif prim == 'grasp' and obs['classmap'][px_y, px_x] != 0:
            held_class, held_id = obs['classmap'][px_y, px_x], obs['idmap'][px_y, px_x]
            executed = sim.pick(position)
            if executed and learned and sim.check_sim():  # hold the object, ask for the placement
                self.holding = True
                mid = self._full_obs(action)
                mid['held_class'], mid['held_id'] = int(held_class), float(held_id)
                mid['heuristic_place'] = get_place_pose(sim, held_class, mid['freespace'])
                self.obs = mid
                info = {'executed': True, 'sorted': False, 'invalid': False, 'place_pix': None, 'relabel': False,
                        'episode_return': self.episode_return, 'actions': self.actions, 'phase': 'place'}
                return self.compact(mid), 0.0, False, False, info
            if executed and not learned:
                freespace = sim.get_observation()['freespace']
                place_pix = get_place_pose(sim, held_class, freespace)
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

        next_obs = self._full_obs(action, executed, place_pix) if not terminated else obs
        prev_relabel = self._relabel_state
        self._relabel_state = self._relabel_status()
        # Hindsight relabelling (paper Fig. 8): all objects are inside *some* target region but not (all) in
        # their own one; reassigning the classes would make this a sorted state.
        relabel = (reward == cfg.reward_step and self._relabel_state[0]
                   and (not prev_relabel[0] or prev_relabel[1] != self._relabel_state[1]))
        info = {'executed': bool(executed), 'sorted': bool(sorted_ and valid), 'invalid': not valid,
                'place_pix': place_pix, 'relabel': bool(relabel), 'episode_return': self.episode_return,
                'actions': self.actions, 'phase': 'act', 'force_stops': sim.force_stops - stops_before}
        self.obs = next_obs
        return self.compact(next_obs), reward, terminated, truncated, info

    # ------------------------------------------------------------------ observations
    def _full_obs(self, last_action, executed=True, place_pix=None):
        """Simulator observation plus (if enabled) the spatial belief. Internal use only (large)."""
        obs = self.sim.get_observation()
        if self.cfg.belief == 'spatial':
            if last_action is not None:
                last_action = dict(last_action, place_pix=place_pix)
            obs['belief_full'] = self.tracker.update(obs, last_action)
        return obs

    def compact(self, obs):
        """What leaves the environment and is stored in the replay buffer (a few 100 KB instead of ~4 MB:
        the raw camera images and the auxiliary maps are dropped)."""
        out = {'colormap': obs['colormap'], 'depthmap': obs['depthmap']}
        if obs.get('held_class'):  # object in the gripper (learned placement): what the place head needs
            out['held_class'] = obs['held_class']
            out['freespace'] = (obs['freespace'] > 0).astype(np.uint8)
            out['heuristic_place'] = obs['heuristic_place']
        if 'belief_full' in obs:
            out['belief'] = compact_belief(obs['belief_full'], self.cfg.belief_age_horizon)
        return out

    # ------------------------------------------------------------------ hindsight relabelling
    def _relabel_status(self):
        possible = self.sim.check_relabel() == 1
        return possible, list(self.sim.new_class) if possible else None

    def relabel(self, obs, next_obs):
        """Compact copies of (obs, next_obs) -- full internal observations -- whose colours (including the
        remembered colours of the spatial belief) use the post-hoc class assignment of the current scene."""
        sim = self.sim
        pair = []
        for o in (obs, next_obs):
            new = self.compact(o)
            if o.get('held_class') and o.get('held_id') in list(sim.id_list):
                new['held_class'] = int(sim.new_class[list(sim.id_list).index(o['held_id'])])
            new['colormap'] = sim.relabel_image([o['colormap']], [o['idmap']], sim.id_list, sim.new_class)[0]
            if 'belief_full' in o:
                b = dict(o['belief_full'])
                b['rgb'] = sim.relabel_image([b['rgb']], [b['id']], sim.id_list, sim.new_class)[0]
                new['belief'] = compact_belief(b, self.cfg.belief_age_horizon)
            pair.append(new)
        return tuple(pair)


def make_env(cfg, gui, num_obj, params, seed=None):
    """Top-level factory (picklable) used by the vectorised environment workers."""
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    return RearrangeEnv(cfg, gui=gui, num_obj=num_obj, params=params)
