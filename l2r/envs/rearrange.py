"""Gym-style wrapper around the PyBullet simulation (`l2r.sim.environment.Environment`).

    env = RearrangeEnv(cfg, gui=False, num_obj=1)
    obs = env.reset()
    obs, reward, terminated, truncated, info = env.step({'primitive': 'push', 'pix': (rot, y, x)})

It owns everything that is not learning: action execution, feasibility checks, the reward (paper Eq. 2),
episode termination and the hindsight relabelling of observations.
"""
import random

import numpy as np
from scipy import ndimage
from skimage.draw import line

from l2r.config import Config
from l2r.learning.belief import BeliefTracker, compact_belief
from l2r.sim.constants import CLASSES, LABELS, TARGET_ORDER
from l2r.sim.environment import Environment


def decode_push(index, push_lengths):
    """Push action index (direction * L + length) -> (angle in rad, length in m). Direction 0/1 = -90/+90 degrees."""
    direction, length = divmod(int(index), len(push_lengths))
    return np.deg2rad(direction * 180.0 - 90), push_lengths[length]


LAST_PRIMITIVES = ('push', 'grasp', 'place', 'pull')
LAST_ACTION_SIZE = len(LAST_PRIMITIVES) + 3  # one-hot primitive, executed, changed fraction of the height map, scene invalid


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
    goal = goal_mask(sim, obj_class)
    if goal is not None:  # mask goals ('grid'): the same rules with the class's goal cells
        fitted = fit_footprint(freespace, *extent)
        candidate = fitted if (fitted is not freespace and (fitted & goal).any()) else clear_of_back_wall(freespace, extent[0])
        return (candidate & goal).astype(candidate.dtype) if (candidate & goal).any() else candidate
    low, high, _ = sim.target_pixels()
    k = [LABELS[name] for name in TARGET_ORDER].index(int(obj_class))
    fitted = fit_footprint(freespace, *extent)
    candidate = fitted if (fitted is not freespace and fitted[low[k]:high[k]].any()) else clear_of_back_wall(freespace, extent[0])
    if not candidate[low[k]:high[k]].any():
        return candidate
    in_region = np.zeros_like(candidate)
    in_region[low[k]:high[k]] = candidate[low[k]:high[k]]  # the heuristic trades row distance against depth: do not let it leave the region
    return in_region


GOAL_INNER_MARGIN = 6  # cells (3 cm): placements keep the object's centre this far inside a goal slot


def corridor_choice(freespace, kept, own):
    """Free space for the heuristic placement with corridor preference. `kept`: free cells whose carry-in corridor is clear; `own`:
    the held object's target region. The first that applies:
    1. clear cells inside the region (`kept & own`);
    2. the unfiltered free space, if the region has free cells but none with a clear corridor (stay in the region);
    3. clear cells elsewhere (`kept`), if the region has no free cell (the best temporary spot);
    4. the unfiltered free space.
    The result has the dtype of `freespace`."""
    free = np.asarray(freespace) > 0
    kept, own = np.asarray(kept, bool), np.asarray(own, bool)
    dtype = np.asarray(freespace).dtype
    if (kept & own).any():
        return (kept & own).astype(dtype)
    if (free & own).any():
        return freespace
    if kept.any():
        return kept.astype(dtype)
    return freespace


def goal_mask(sim, obj_class):
    """(H, W) bool goal cells of the class with label `obj_class` for mask layouts, or None for the 'bands' layout."""
    scene = getattr(sim, 'scene', None)
    if scene is None or getattr(scene, 'layout', 'bands') == 'bands':
        return None
    name = TARGET_ORDER[[LABELS[n] for n in TARGET_ORDER].index(int(obj_class))]
    return scene.goal_masks()[name]


def get_place_pose(sim, obj_class, freespace):
    """Heuristic placement pixel (col, row) for an object of class label `obj_class`."""
    goal = goal_mask(sim, obj_class)
    if goal is not None:  # free cell well inside the goal, nearest to its centre (a slot has edges on all sides, unlike a band)
        free = (np.asarray(freespace) > 0)
        inner = ndimage.binary_erosion(goal, iterations=GOAL_INNER_MARGIN)
        inner = inner if (free & inner).any() else goal
        cells = np.argwhere(free & inner) if (free & inner).any() else np.argwhere(free)
        if not len(cells):
            rows, cols = np.nonzero(goal)
            return int(np.median(cols)), int(np.median(rows))
        g_rows, g_cols = np.nonzero(goal)
        score = np.hypot(cells[:, 0] - g_rows.mean(), cells[:, 1] - g_cols.mean())
        row, col = cells[int(np.argmin(score))]
        return int(col), int(row)
    y_down, y_up, _ = sim.target_pixels()  # in TARGET_ORDER
    index = [LABELS[name] for name in TARGET_ORDER].index(int(obj_class))
    return select_placement(y_up[index], y_down[index], freespace)


PULL_WALL_ZONE = 0.07  # m: a pull of an object this close to a side wall drags it away from that wall
PULL_APPROACH = np.radians(30)  # in from the cabinet's middle towards the wall
PULL_LENGTH = 0.08  # m


def default_pull(position, ws):
    """Parameters of a pull at `position` (x, y) when the action gives none: next to a side wall, in from the middle at 30 degrees
    and dragged 8 cm away from the wall and towards the front (45 degrees); elsewhere straight in and straight towards the front."""
    b = ws.bounds
    y = float(position[1])
    if min(y - b[1][0], b[1][1] - y) < PULL_WALL_ZONE:
        side = np.sign(y) or 1.0
        return {'approach': float(-side * PULL_APPROACH), 'heading': float(np.arctan2(-side, 1.0)), 'length': PULL_LENGTH}
    return {'approach': 0.0, 'heading': 0.0, 'length': PULL_LENGTH}


def make_simulator(params):
    """The simulator class of `params.backend` (PyBullet by default; MuJoCo is imported only when asked for)."""
    backend = getattr(params, 'backend', 'pybullet')
    if backend == 'mujoco':
        from l2r.sim.mj.native import NativeEnvironment
        return NativeEnvironment
    if backend == 'mujoco_urdf':
        from l2r.sim.mj.environment import MujocoEnvironment
        return MujocoEnvironment
    return Environment


class RearrangeEnv:
    def __init__(self, cfg=None, gui=False, num_obj=1, params=None):
        self.cfg = cfg or Config()
        self.num_obj = num_obj
        self.sim = make_simulator(params)(gui=gui, num_obj=[num_obj], params=params)
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
        self._last_action = np.zeros(LAST_ACTION_SIZE, np.float32)  # nothing done yet
        self._last_pix = np.array([-1, -1], np.int16)
        self._relabel_state = self._relabel_status()
        return self.compact(self.obs)

    def step(self, action):
        """action: {'primitive': 'push'|'grasp'|'place'|'pull', 'pix': (rotation_index, row, col)} in Q-map coordinates.

        Parameterised primitives (optional keys, see `l2r.sim.primitives`): 'push' with 'heading' (rad, any floor direction
        away from the robot) and 'length' (m) instead of the encoded direction / length; 'grasp' with 'approach' (rad, the
        approach line rotated about the vertical); 'pull' (attach at the pixel, drag 'length' m along 'heading', default
        towards the robot, release) with an optional 'approach'.

        placement='heuristic': 'grasp' picks and places in one action (placement by rule).
        placement='learned':   'grasp' only picks. While the object is held, the observation (taken after the
                               pick, with the held class and the free cells) is returned with reward 0 and the
                               next action must be 'place'; the reward is given for the complete pick-and-place.
        """
        obs, sim, cfg = self.obs, self.sim, self.cfg
        prim, pix = action['primitive'], action['pix']
        # parameters carried in the action tuple (object agent with parameterised skills, `--skill_params`):
        # push (0, row, col, length fraction, heading rad), grasp (0, row, col, approach rad)
        if prim == 'push' and len(pix) >= 5 and 'heading' not in action:
            low, high = cfg.push_length_range
            action = dict(action, heading=float(pix[4]), length=low + float(pix[3]) * (high - low))
        elif prim == 'grasp' and len(pix) >= 4 and 'approach' not in action:
            action = dict(action, approach=float(pix[3]))
        px_x, px_y = int(pix[2]), int(pix[1])
        position = self.ws.px_to_xy(px_x, px_y)
        executed, place_pix, attempted = False, None, False
        stops_before = sim.force_stops
        if not self.holding:  # the snapshot is taken at the start of the (pick-and-place) action
            self._positions = sim.scene.positions()
        learned = cfg.placement == 'learned'
        if self.holding and prim != 'place':
            raise ValueError("an object is held: the next action must be 'place'")

        if prim == 'push' and 'heading' in action:  # parameterised push: any floor direction (rad), length in m
            attempted = True
            executed = sim.push(position, push_length=float(action.get('length', max(cfg.push_lengths))), heading=float(action['heading']))
        elif prim == 'pull':  # attach (cup / jaws), drag along the floor, release
            attempted = True
            default = default_pull(position, self.ws)  # a bare pull (the object agent's slot, the expert): off the nearer wall
            executed = sim.pull(position, approach=float(action.get('approach', default['approach'])),
                                heading=float(action.get('heading', default['heading'])),
                                distance=float(action.get('length', default['length'])))
        elif prim == 'push':
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
            executed = sim.pick(position, yaw=cfg.jaw_angle(pix[0]) if cfg.end_effector == 'gripper' else 0.0,
                                approach=float(action['approach']) if 'approach' in action else self._approach_for(pix),
                                height=None if action.get('height') is None else float(action['height']))
            if executed and learned and sim.is_valid():  # hold the object, ask for the placement
                self.holding = True
                mid = self._full_obs(action)
                if cfg.footprint_placement:
                    mid['freespace'] = self._placement_space(held_class, mid['freespace'], held_extent)
                mid['held_class'], mid['held_id'] = int(held_class), float(held_object)
                mid['heuristic_place'] = get_place_pose(sim, held_class, self._corridor_free(mid['freespace'], mid['depthmap'], held_extent, held_class))
                self.obs = mid
                info = {'executed': True, 'attempted': True, 'sorted': False, 'invalid': False, 'place_pix': None, 'relabel': False,
                        'episode_return': self.episode_return, 'actions': self.actions, 'phase': 'place'}
                self._record_last(prim, pix, True, obs, mid)
                return self.compact(mid), 0.0, False, False, info
            if executed and not learned:
                view = sim.observe()  # one render for both maps
                freespace = view['freespace']
                if cfg.footprint_placement:
                    freespace = self._placement_space(held_class, freespace, held_extent)
                place_pix = get_place_pose(sim, held_class, self._corridor_free(freespace, view['depthmap'], held_extent, held_class))
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
        relabel_reward = self._relabel_reward() if relabel else None
        info = {'executed': bool(executed), 'attempted': attempted, 'sorted': bool(sorted_ and valid), 'invalid': not valid,
                'place_pix': place_pix, 'relabel': bool(relabel), 'episode_return': self.episode_return,
                **({'relabel_reward': relabel_reward} if relabel_reward is not None else {}),
                'actions': self.actions, 'phase': 'act', 'force_stops': sim.force_stops - stops_before,
                'shaping': shaping, 'refused': bool(getattr(sim, 'last_refused', False))}
        self.obs = next_obs
        # a terminal step returns the previous observation: the outcome record compares with what the scene looks like now
        self._record_last(prim, pix, executed, obs, sim.observe() if terminated else next_obs, invalid=not valid)
        return self.compact(next_obs), reward, terminated, truncated, info

    def _corridor_free(self, freespace, depthmap, extent, held_class):
        """With cfg.place_corridor: the free space offered to the heuristic placement, preferring cells whose carry-in corridor holds
        no other object, without ever trading the object's own target region for a clear corridor (see `corridor_choice`)."""
        if not self.cfg.place_corridor:
            return freespace
        from l2r.envs.approach import place_corridor_obstacles
        o = self.cfg.offset_px
        depth = np.asarray(depthmap)[o:np.asarray(depthmap).shape[0] - o, o:]
        width = int(extent[1]) if extent is not None and len(extent) > 1 and extent[1] else 16
        free = np.asarray(freespace)
        clear = place_corridor_obstacles(depth, width // 2 + 2) == 0
        kept = (free > 0) & clear[:free.shape[0], :free.shape[1]]
        return corridor_choice(freespace, kept, self._own_region(held_class, free.shape))

    def _own_region(self, held_class, shape):
        """(shape) bool: the target region of the class with label `held_class` (goal mask, or the band rows)."""
        goal = goal_mask(self.sim, held_class)
        if goal is not None:
            return np.asarray(goal, bool)[:shape[0], :shape[1]]
        y_down, y_up, _ = self.sim.target_pixels()
        index = [LABELS[name] for name in TARGET_ORDER].index(int(held_class))
        low, high = sorted((int(y_down[index]), int(y_up[index])))
        own = np.zeros(shape, bool)
        own[max(low, 0):max(high, 0)] = True
        return own

    def _approach_for(self, pix):
        """Approach angle of a grasp without an explicit one: straight in, or (cfg.grasp_approach == 'clear') the clearest line in
        the height map (`l2r.envs.approach`)."""
        if self.cfg.grasp_approach != 'clear':
            return 0.0
        from l2r.envs.approach import choose_approach
        return choose_approach(self.obs, (int(pix[1]), int(pix[2])))[0]

    def _placement_space(self, held_class, freespace, extent):
        if self.cfg.placement_space == 'region':
            return placement_space(self.sim, held_class, freespace, extent)
        return clear_of_back_wall(freespace, extent[0])

    # ------------------------------------------------------------------ reward variants
    def _potential(self):
        """Shaping potential phi(s) of the current scene (0 for the sparse reward); 0 at a goal state."""
        if self.cfg.reward_mode not in ('progress', 'distance', 'combined'):
            return 0.0
        return self._potential_from(*self.sim.scene.progress())

    def _relabel_reward(self):
        """Reward of the hindsight-relabelled goal transition with shaping (`cfg.relabel_shaping`): reward_goal - phi of the state
        BEFORE the action under the RELABELLED classes, which is what a real final step earns. None when it does not apply."""
        cfg, scene = self.cfg, self.sim.scene
        if not cfg.relabel_shaping or cfg.reward_mode not in ('progress', 'distance', 'combined') or self._positions is None:
            return None
        names = {label: name for name, label in LABELS.items()}
        relabelled = dict(zip(scene.relabel_ids, scene.relabel_classes))
        distances = []
        for object_id, position in zip(scene.object_ids, self._positions):
            name = names[relabelled[object_id]] if object_id in relabelled else scene.class_name(object_id)
            distances.append(scene.goal_distance(position, name))
        fraction = float(np.mean([d == 0.0 for d in distances]))
        return cfg.reward_goal - self._potential_from(fraction, float(np.mean(distances)))

    def _potential_from(self, fraction, distance):
        """The potential of the reward mode from the sorted fraction and the mean distance to the goals."""
        cfg = self.cfg
        width = float(self.ws.bounds[1][1] - self.ws.bounds[1][0])
        if cfg.reward_mode == 'progress':
            return cfg.reward_shaping_scale * (fraction - 1.0)
        if cfg.reward_mode == 'distance':
            return -cfg.reward_shaping_scale * distance / width
        return cfg.reward_shaping_scale * ((fraction - 1.0) - distance / width)  # combined: placements and partial progress

    def _shaped(self, reward, terminated, valid):
        """Add the reward variant's extra term to the sparse reward; returns (reward, extra term)."""
        cfg, extra = self.cfg, 0.0
        if cfg.reward_mode in ('progress', 'distance', 'combined'):
            phi_next = self._potential() if (valid and not terminated) else 0.0  # absorbing states have potential 0
            extra = cfg.gamma * phi_next - self._phi
            self._phi = phi_next
        if cfg.reward_mode in ('noop', 'combined') and not terminated:
            moved = np.abs(self.sim.scene.positions() - self._positions).max() if len(self._positions) else 0.0
            extra += -cfg.noop_penalty if moved < 0.005 else 0.0
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

    def _goal_map(self):
        """(H, W) uint8 goal of every cell: 1 + CLASSES index of the class whose goal it is, 0 elsewhere (any layout)."""
        out = np.zeros(self.ws.heightmap_shape, np.uint8)
        scene = self.sim.scene
        if not hasattr(scene, 'goal_masks'):  # a backend without goal masks: the band rows
            for k, (low, high) in enumerate(self._regions()):
                out[max(int(low), 0):max(int(high), 0)] = k + 1
            return out
        for k, name in enumerate(CLASSES):
            out[scene.goal_masks()[name]] = k + 1
        return out

    def _regions(self):
        """(3, 2) rows [low, high] of the target region of each class (CLASSES order) in the un-padded heightmap."""
        return np.array([[self.ws.y_to_row(y) for y in self.sim.scene.regions[name]] for name in CLASSES], np.int16)

    def _record_last(self, primitive, pix, executed, before, after, invalid=False):
        """What the agent may know about its previous action (no privileged information: the robot knows what it did, whether the
        primitive completed, and can compare two height maps): primitive one-hot, executed, fraction of the cabinet's cells whose height
        changed by more than 1 cm, the scene became invalid; and the action pixel. The same record in the NEXT observation is the
        label of the outcome heads (`object_trainer`)."""
        out = np.zeros(LAST_ACTION_SIZE, np.float32)
        out[LAST_PRIMITIVES.index(primitive) if primitive in LAST_PRIMITIVES else 0] = 1.0
        out[len(LAST_PRIMITIVES)] = float(bool(executed))
        out[len(LAST_PRIMITIVES) + 1] = float((np.abs(np.asarray(after['depthmap']) - np.asarray(before['depthmap'])) > 0.01).mean())
        out[len(LAST_PRIMITIVES) + 2] = float(bool(invalid))
        self._last_action, self._last_pix = out, np.array([pix[1], pix[2]], np.int16)

    def compact(self, obs):
        """What leaves the environment and is stored in the replay buffer (a few 100 KB instead of ~4 MB:
        the raw camera images and the auxiliary maps are dropped)."""
        out = {'colormap': obs['colormap'], 'depthmap': obs['depthmap'], 'regions': self._regions(), 'goal_map': self._goal_map(),
               'last_action': getattr(self, '_last_action', np.zeros(LAST_ACTION_SIZE, np.float32)),
               'last_pix': getattr(self, '_last_pix', np.array([-1, -1], np.int16))}
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
