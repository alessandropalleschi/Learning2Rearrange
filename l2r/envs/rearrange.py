"""Gym-style wrapper around the PyBullet simulation (`l2r.sim.environment.Environment`).

    env = RearrangeEnv(cfg, gui=False, num_obj=1)
    obs = env.reset()
    obs, reward, terminated, truncated, info = env.step({'primitive': 'push', 'pix': (rot, y, x)})

It owns everything that is not learning: action execution, feasibility checks, the reward (paper Eq. 2),
episode termination and the hindsight relabelling of observations.
"""
import random

import numpy as np
from skimage.draw import line

from l2r.config import Config
from l2r.learning.belief import BeliefTracker, compact_belief
from l2r.sim.constants import CLASSES, LABELS, TARGET_ORDER
from l2r.sim.environment import Environment


def decode_push(index, push_lengths):
    """Push action index (direction * L + length) -> (angle in rad, length in m). Direction 0/1 = -90/+90 degrees."""
    direction, length = divmod(int(index), len(push_lengths))
    return np.deg2rad(direction * 180.0 - 90), push_lengths[length]


def check_push(ws, depth, pix_x, pix_y, angle, push_length=0.08):
    """True if there is something in the way of the push (there must be an object to push).
    `ws` is the `Workspace` (pixel <-> metre conversion); pixel coordinates are those of the Q-maps."""
    y_end = ws.px_to_xy(pix_x, pix_y)[1] + push_length * np.sin(angle)
    depth_new = depth[4:depth.shape[0] - 4, 4:depth.shape[1]]
    end_px = min(ws.y_to_row(y_end), depth_new.shape[0] - 1)
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


def object_depth_px(obs, object_id):
    """Extent (pixels along x, the cabinet depth) of an object in the id map; 0 if it is not visible."""
    cols = np.nonzero(obs['idmap'] == object_id)[1]
    return int(cols.max() - cols.min() + 1) if cols.size else 0


def object_extent_px(obs, object_id):
    """(depth, width) in cells of an object in the top view: its extent along x (cabinet depth) and along y (width)."""
    rows, cols = np.nonzero(obs['idmap'] == object_id)
    if cols.size == 0:
        return 0, 0
    return int(cols.max() - cols.min() + 1), int(rows.max() - rows.min() + 1)


def fit_footprint(freespace, depth_px, width_px, margin_cols=4, margin_rows=2):
    """Cells where the whole footprint of the held object fits into free space.

    The placement cell is where the cup holds the object's front face, centred on its width: the footprint covers `depth_px`
    (+ margin) cells behind the cell and `width_px` (+ margin) rows around it. Outside the map counts as occupied, which also keeps
    the object off the back and side walls. If nothing fits the plain free-space map is returned (the object may still fit,
    for example by being pushed against others)."""
    import cv2
    columns, rows = depth_px + margin_cols, width_px + 2 * margin_rows
    kernel = np.ones((rows, columns), np.uint8)
    fit = cv2.erode((freespace > 0).astype(np.uint8), kernel, anchor=(columns - 1, rows // 2),
                    borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return fit.astype(freespace.dtype) if fit.any() else freespace


def clear_of_back_wall(freespace, depth_px, margin=4):
    """Free-space map without the cells closer to the back wall than the object is deep (+ a margin): an object
    placed there would hit the wall, because the tool holds it at its front face."""
    out = freespace.copy()
    out[:, :depth_px + margin] = 0
    return out


def placement_space(sim, obj_class, freespace, extent):
    """Free-space map offered to the placement for a held object of class label `obj_class` and top-view extent (depth, width).

    Only cells inside the object's own target region are offered whenever there are any (an object put elsewhere can not be
    sorted, and the placement heuristic trades row distance against depth, so it would otherwise pick cells outside the region).
    Preferably only cells where the whole footprint fits (`fit_footprint`), but only if that leaves cells inside that
    region: otherwise the nearest valid cell lies in another region, the object is put there, picked up again and the
    episode ping-pongs (measured: expert success fell from 87% to 33% at 4 objects). Then it falls back to keeping only the
    back wall clear (`clear_of_back_wall`)."""
    low, high, _ = sim.target_pixels()
    k = [LABELS[name] for name in TARGET_ORDER].index(int(obj_class))
    fitted = fit_footprint(freespace, *extent)
    candidate = fitted if (fitted is not freespace and fitted[low[k]:high[k]].any()) else clear_of_back_wall(freespace, extent[0])
    if not candidate[low[k]:high[k]].any():
        return candidate
    in_region = np.zeros_like(candidate)
    in_region[low[k]:high[k]] = candidate[low[k]:high[k]]  # the heuristic trades row distance against depth: do not let it leave the region
    return in_region


def get_place_pose(sim, obj_class, freespace):
    """Heuristic placement pixel (col, row) for an object of class label `obj_class`."""
    y_down, y_up, _ = sim.target_pixels()  # in TARGET_ORDER
    index = [LABELS[name] for name in TARGET_ORDER].index(int(obj_class))
    return select_placement(y_up[index], y_down[index], freespace)


class RearrangeEnv:
    def __init__(self, cfg=None, gui=False, num_obj=1, params=None):
        self.cfg = cfg or Config()
        self.num_obj = num_obj
        self.sim = Environment(gui=gui, num_obj=[num_obj], params=params)
        self.ws = self.sim.ws
        self.obs = None  # full (internal) observation of the current state
        self.tracker = BeliefTracker(self.cfg.belief_age_horizon)
        self.actions = 0
        self.episode_return = 0.0
        self.holding = False
        self._phi = 0.0  # shaping potential of the current state
        self._positions = None  # object positions before the action (no-op detection)

    # ------------------------------------------------------------------ gym API
    def reset(self, num_obj=None, seed=None):
        if seed is not None:  # reproducible scene (object sizes, positions, target regions)
            np.random.seed(seed)
            random.seed(seed)
        if num_obj is not None:
            self.num_obj = num_obj
            self.sim.num_obj = [num_obj]
        while True:  # resample until the scene is valid and not already sorted
            if self.sim.reset() and self.sim.fraction_sorted() < 1:
                break
        self.actions = 0
        self.episode_return = 0.0
        self.holding = False
        self.tracker.reset()
        self._phi = self._potential()
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
        position = self.ws.px_to_xy(px_x, px_y)
        executed, place_pix, attempted = False, None, False
        stops_before = sim.force_stops
        if not self.holding:  # the snapshot is taken at the start of the (pick-and-place) action
            self._positions = sim.scene.positions()
        learned = cfg.placement == 'learned'
        if self.holding and prim != 'place':
            raise ValueError("an object is held: the next action must be 'place'")

        if prim == 'push':
            angle, length = decode_push(pix[0], cfg.push_lengths)
            if cfg.hierarchy:  # continuous length: pix[3] in 0..1 of the range
                length = cfg.push_length_range[0] + pix[3] * (cfg.push_length_range[1] - cfg.push_length_range[0])
            if check_push(self.ws, obs['depthmap'], px_x, px_y, angle, length):
                attempted = True
                executed = sim.push(position, angle, push_length=length)
        elif prim == 'place':
            attempted = True
            place_pix = (px_x, px_y)  # (col, row) of the free cell
            executed = sim.place(position)
            self.holding = False
        elif prim == 'grasp' and obs['classmap'][px_y, px_x] != 0:
            attempted = True
            held_class, held_id = obs['classmap'][px_y, px_x], obs['idmap'][px_y, px_x]
            held_extent = object_extent_px(obs, held_id) if cfg.footprint_placement else (0, 0)
            held_object = obs.get('instance_to_object', {}).get(int(held_id), held_id)  # ground-truth id (relabelling)
            executed = sim.pick(position, yaw=cfg.jaw_angle(pix[0]) if cfg.end_effector == 'gripper' else 0.0)
            if executed and learned and sim.is_valid():  # hold the object, ask for the placement
                self.holding = True
                mid = self._full_obs(action)
                if cfg.footprint_placement:
                    mid['freespace'] = self._placement_space(held_class, mid['freespace'], held_extent)
                mid['held_class'], mid['held_id'] = int(held_class), float(held_object)
                mid['heuristic_place'] = get_place_pose(sim, held_class, mid['freespace'])
                self.obs = mid
                info = {'executed': True, 'attempted': True, 'sorted': False, 'invalid': False, 'place_pix': None, 'relabel': False,
                        'episode_return': self.episode_return, 'actions': self.actions, 'phase': 'place'}
                return self.compact(mid), 0.0, False, False, info
            if executed and not learned:
                freespace = sim.observe()['freespace']
                if cfg.footprint_placement:
                    freespace = self._placement_space(held_class, freespace, held_extent)
                place_pix = get_place_pose(sim, held_class, freespace)
                executed = sim.place(self.ws.px_to_xy(*place_pix))

        sim.wait_static()
        sorted_ = sim.fraction_sorted() == 1
        valid = sim.is_valid()
        self.actions += 1

        if not valid:
            reward = cfg.reward_invalid
        elif sorted_:
            reward = cfg.reward_goal
        else:
            reward = cfg.reward_step
        terminated = sorted_ or not valid
        base_reward = reward  # hindsight relabelling looks at the unshaped reward
        reward, shaping = self._shaped(reward, terminated, valid)
        truncated = (not terminated) and self.actions >= cfg.max_actions
        self.episode_return += reward

        next_obs = self._full_obs(action, executed, place_pix) if not terminated else obs
        prev_relabel = self._relabel_state
        self._relabel_state = self._relabel_status()
        # Hindsight relabelling (paper Fig. 8): all objects are inside *some* target region but not (all) in
        # their own one; reassigning the classes would make this a sorted state.
        relabel = (base_reward == cfg.reward_step and self._relabel_state[0]
                   and (not prev_relabel[0] or prev_relabel[1] != self._relabel_state[1]))
        info = {'executed': bool(executed), 'attempted': attempted, 'sorted': bool(sorted_ and valid), 'invalid': not valid,
                'place_pix': place_pix, 'relabel': bool(relabel), 'episode_return': self.episode_return,
                'actions': self.actions, 'phase': 'act', 'force_stops': sim.force_stops - stops_before,
                'shaping': shaping}
        self.obs = next_obs
        return self.compact(next_obs), reward, terminated, truncated, info

    def _placement_space(self, held_class, freespace, extent):
        if self.cfg.placement_space == 'region':
            return placement_space(self.sim, held_class, freespace, extent)
        return clear_of_back_wall(freespace, extent[0])

    # ------------------------------------------------------------------ reward variants
    def _potential(self):
        """Shaping potential phi(s) of the current scene (0 for the sparse reward); 0 at a goal state."""
        cfg = self.cfg
        if cfg.reward_mode not in ('progress', 'distance'):
            return 0.0
        fraction, distance = self.sim.scene.progress()
        if cfg.reward_mode == 'progress':
            return cfg.reward_shaping_scale * (fraction - 1.0)
        return -cfg.reward_shaping_scale * distance / float(self.ws.bounds[1][1] - self.ws.bounds[1][0])

    def _shaped(self, reward, terminated, valid):
        """Add the reward variant's extra term to the sparse reward; returns (reward, extra term)."""
        cfg, extra = self.cfg, 0.0
        if cfg.reward_mode in ('progress', 'distance'):
            phi_next = self._potential() if (valid and not terminated) else 0.0  # absorbing states have potential 0
            extra = cfg.gamma * phi_next - self._phi
            self._phi = phi_next
        elif cfg.reward_mode == 'noop' and not terminated:
            moved = np.abs(self.sim.scene.positions() - self._positions).max() if len(self._positions) else 0.0
            extra = -cfg.noop_penalty if moved < 0.005 else 0.0
        return reward + extra, extra

    # ------------------------------------------------------------------ observations
    def _full_obs(self, last_action, executed=True, place_pix=None):
        """Simulator observation plus (if enabled) the spatial belief. Internal use only (large)."""
        obs = self.sim.observe()
        if self.cfg.belief == 'spatial':
            if last_action is not None:
                last_action = dict(last_action, place_pix=place_pix)
            obs['belief_full'] = self.tracker.update(obs, last_action)
        return obs

    def _regions(self):
        """(3, 2) rows [low, high] of the target region of each class (CLASSES order) in the un-padded heightmap."""
        return np.array([[self.ws.y_to_row(y) for y in self.sim.scene.regions[name]] for name in CLASSES], np.int16)

    def compact(self, obs):
        """What leaves the environment and is stored in the replay buffer (a few 100 KB instead of ~4 MB:
        the raw camera images and the auxiliary maps are dropped)."""
        out = {'colormap': obs['colormap'], 'depthmap': obs['depthmap'], 'regions': self._regions()}
        if obs.get('held_class'):  # object in the gripper (learned placement): what the place head needs
            out['held_class'] = obs['held_class']
            out['freespace'] = (obs['freespace'] > 0).astype(np.uint8)
            out['heuristic_place'] = obs['heuristic_place']
        if 'belief_full' in obs:
            out['belief'] = compact_belief(obs['belief_full'], self.cfg.belief_age_horizon)
        return out

    # ------------------------------------------------------------------ hindsight relabelling
    def _relabel_status(self):
        possible = self.sim.relabel_candidates() == 1
        return possible, list(self.sim.scene.relabel_classes) if possible else None

    def _relabel_instances(self, obs):
        """(instance ids in obs['idmap'], new classes) of the objects to recolour. With a learned perception the instance
        ids are not object ids: they are translated with the privileged `instance_to_object` match."""
        gt = dict(zip(self.sim.scene.relabel_ids, self.sim.scene.relabel_classes))
        mapping = obs.get('instance_to_object')
        if mapping is None:
            return list(gt), list(gt.values())
        pairs = [(instance, gt[uid]) for instance, uid in mapping.items() if uid in gt]
        return [p[0] for p in pairs], [p[1] for p in pairs]

    def relabel(self, obs, next_obs):
        """Compact copies of (obs, next_obs) -- full internal observations -- whose colours (including the
        remembered colours of the spatial belief) use the post-hoc class assignment of the current scene."""
        gt_ids, gt_classes = list(self.sim.scene.relabel_ids), list(self.sim.scene.relabel_classes)
        pair = []
        for o in (obs, next_obs):
            new = self.compact(o)
            if o.get('held_class') and o.get('held_id') in gt_ids:
                new['held_class'] = int(gt_classes[gt_ids.index(o['held_id'])])
            ids, classes = self._relabel_instances(o)
            new['colormap'] = self.sim.relabel_colormap(o['colormap'], o['idmap'], ids, classes)
            if 'belief_full' in o:
                b = dict(o['belief_full'])
                b['rgb'] = self.sim.relabel_colormap(b['rgb'], b['id'])
                new['belief'] = compact_belief(b, self.cfg.belief_age_horizon)
            pair.append(new)
        return tuple(pair)


def make_env(cfg, gui, num_obj, params, seed=None):
    """Top-level factory (picklable) used by the vectorised environment workers."""
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    return RearrangeEnv(cfg, gui=gui, num_obj=num_obj, params=params)
