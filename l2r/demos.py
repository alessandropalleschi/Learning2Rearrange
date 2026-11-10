"""Scripted expert and demonstration recording.

The expert uses privileged information (object ids and poses of the simulator) to choose, from the same observation the
policy sees, which object to move: a misplaced, visible object, front-most first, picked at its front face and placed by the
footprint-aware heuristic (or by the learned placement action when `placement == 'learned'`). It pushes only when nothing misplaced is
reachable. Its successful and unsuccessful transitions are recorded exactly like the agent's (the environment gives rewards and
the hindsight relabelling), so they can seed the replay buffer and pre-train the Q-network (DQfD: TD + margin loss).

    python scripts/collect_demos.py --episodes 100 --objects 1 5 --out data/demos/expert.pkl --placement learned --footprint_placement
"""
import pickle

import numpy as np

from l2r.envs.rearrange import check_push, decode_push


GRASP_DEPTH_CELLS = 2  # cells (5 mm each) behind the front-most visible cell of the object


class ScriptedExpert:
    """Chooses (primitive, pix) for a `RearrangeEnv` from its full observation and the simulator state."""

    MAX_FAILURES = 2  # grasp attempts on one object that did not hold before the expert tries something else
    NEAR_MARGIN = 0.02  # m: a misplaced object closer to its region than the longest push minus this is pushed into it
    MAX_EXPOSE_PUSHES = 2  # pushes of one object to uncover hidden ones

    CONTINUOUS_LENGTHS = (0.04, 0.08, 0.12)  # m: the lengths the expert chooses from when the environment's push length is continuous

    def __init__(self, env, push_lengths=None):
        """push_lengths: the selectable lengths (default: the environment's; with a continuous length (`cfg.hierarchy`)
        CONTINUOUS_LENGTHS, and the chosen one is returned as a fraction of `cfg.push_length_range` in pix[3])."""
        self.env = env
        base = env.cfg
        self.continuous = bool(getattr(base, 'hierarchy', False))
        if push_lengths is None and self.continuous:
            push_lengths = self.CONTINUOUS_LENGTHS
        if push_lengths is not None:
            import dataclasses
            base = dataclasses.replace(base, push_lengths=tuple(push_lengths), hierarchy=False)
        self.cfg = base
        self.failures = {}  # object id -> failed grasps
        self.pushes = {}  # object id -> pushes made on it

    def object_at(self, pix):
        return int(self.env.obs['idmap'][pix[1], pix[2]])

    def report(self, primitive, pix, object_id, executed):
        if primitive == 'grasp' and not executed:
            self.failures[object_id] = self.failures.get(object_id, 0) + 1

    def _misplaced_visible(self):
        env, scene = self.env, self.env.sim.scene
        idmap, found = env.obs['idmap'], []
        for object_id, (pos, _) in scene.poses().items():
            if scene.in_target(pos[1], scene.names[object_id]):
                continue
            rows, cols = np.nonzero(idmap == object_id)
            if rows.size >= 6:  # visible enough to touch
                aim_row, aim_col = self._aim_cell(rows, cols)
                found.append((pos[0], object_id, aim_row, aim_col))
        return sorted(found, reverse=True)  # largest x first: nothing is in front of it

    @staticmethod
    def _aim_cell(rows, cols):
        """A cell that belongs to the object, GRASP_DEPTH_CELLS behind its front-most cell (or the deepest cell there is).

        The approach stops on contact, but the suction only engages on a real contact point and the surface at cup height is often
        a few mm behind the front-most cell of the top view (tapered bottles). The cell must be one of the object's own cells: the
        environment ignores a grasp whose pixel is not on an object."""
        front = cols.max()
        col = front - GRASP_DEPTH_CELLS
        while not (cols == col).any() and col > cols.min():
            col -= 1
        at_col = rows[cols == col] if (cols == col).any() else rows[cols == cols.min()]
        row = int(at_col[np.argmin(np.abs(at_col - np.median(at_col)))])  # the object's own cell closest to the middle
        return row, int(col if (cols == col).any() else cols.min())

    def act(self):
        """(primitive, pix) or None if the expert has nothing sensible to do. A push is (index, row, col) with index = direction * L +
        length index, or (direction, row, col, fraction of the length range) if the environment's push length is continuous."""
        choice = self._act()
        if choice is None or choice[0] != 'push' or not self.continuous:
            return choice
        index, row, col = choice[1]
        lengths, (low, high) = self.cfg.push_lengths, self.env.cfg.push_length_range
        fraction = (lengths[index % len(lengths)] - low) / (high - low)
        return 'push', (index // len(lengths), row, col, float(np.clip(fraction, 0.0, 1.0)))

    def _act(self):
        env, cfg = self.env, self.cfg
        if env.holding:
            col, row = env.obs['heuristic_place']
            return 'place', (0, int(row), int(col))
        misplaced = self._misplaced_visible()
        candidates = [c for c in misplaced if self.failures.get(c[1], 0) < self.MAX_FAILURES]
        if candidates:
            _, object_id, row, col = candidates[0]
            if self._near_region(object_id):  # one push finishes it: no pick and place
                action = self._push_towards_target(object_id, cfg)
                if action:
                    self.pushes[object_id] = self.pushes.get(object_id, 0) + 1
                    return action
            return 'grasp', (0, row, col)
        # nothing can be grasped: push a misplaced object (the one that resisted most) towards its target region
        for _, object_id, _, _ in sorted(misplaced, key=lambda c: -self.failures.get(c[1], 0)):
            action = self._push_towards_target(object_id, cfg)
            if action:
                return action
        # every misplaced object is hidden behind others: push the front-most object aside to expose them
        action = self._expose_hidden(cfg)
        return action or self._push_blocker(cfg)

    def _near_region(self, object_id):
        """True if the object lies outside its target region but within one push (NEAR_REGION m) of it, and was not pushed
        before (a push that did not work is followed by a grasp)."""
        scene = self.env.sim.scene
        low, high = scene.regions[scene.class_name(object_id)]
        y = scene.poses()[object_id][0][1]
        distance = max(low - y, y - high, 0.0)
        reach = max(self.cfg.push_lengths) - self.NEAR_MARGIN  # 0.06 m for the single 8 cm push of the paper
        return 0.0 < distance <= reach and self.pushes.get(object_id, 0) < 1

    def _expose_hidden(self, cfg):
        """Push the front-most visible object (at most MAX_EXPOSE_PUSHES times each) towards the middle of its target region."""
        scene, obs = self.env.sim.scene, self.env.obs
        visible = []
        for object_id, (pos, _) in scene.poses().items():
            if (obs['idmap'] == object_id).sum() >= 6 and self.pushes.get(object_id, 0) < self.MAX_EXPOSE_PUSHES:
                visible.append((pos[0], object_id))
        for _, object_id in sorted(visible, reverse=True):
            action = self._push_towards_target(object_id, cfg)
            if action:
                self.pushes[object_id] = self.pushes.get(object_id, 0) + 1
                return action
        return None

    def _push_towards_target(self, object_id, cfg):
        """Blade next to the object on the side away from its target region, moving towards the region."""
        env, obs, scene = self.env, self.env.obs, self.env.sim.scene
        rows, cols = np.nonzero(obs['idmap'] == object_id)
        if rows.size == 0:
            return None
        low, high = scene.regions[scene.class_name(object_id)]
        y = scene.poses()[object_id][0][1]
        direction = 1 if (low + high) / 2 > y else 0  # 1: towards +y, 0: towards -y
        index = direction * len(cfg.push_lengths) + self._length_index(y, low, high, direction, cfg.push_lengths)
        angle, length = decode_push(index, cfg.push_lengths)
        height = obs['classmap'].shape[0]
        for gap in (3, 4, 2, 5):
            row = int(np.clip(rows.min() - gap if direction == 1 else rows.max() + gap, 0, height - 1))
            col = int(np.median(cols))
            if check_push(env.ws, obs['depthmap'], col, row, angle, length) and self._blade_touches(obs, row, col, direction, length):
                return 'push', (index, row, col)
        return None

    @staticmethod
    def _length_index(y, low, high, direction, lengths):
        """Index of the push length that moves an object at `y` closest to the middle of its region [low, high] (a push moves it by
        about its length). With a single length that is index 0 (the paper's 8 cm)."""
        sign = 1.0 if direction == 1 else -1.0
        landing = [abs((y + sign * length) - (low + high) / 2) for length in lengths]
        return int(np.argmin(landing))

    @staticmethod
    def _blade_touches(obs, row, col, direction, length):
        """The blade's own strip (+-2 cells around its column) has object cells in its path (not only somewhere in front of it)."""
        from l2r.learning.masks import BLADE_HALF_WIDTH_CELLS, PUSH_OBJECT_HEIGHT, push_path_cells
        crop = obs['depthmap'][4:-4, 4:]
        cells = push_path_cells(crop > PUSH_OBJECT_HEIGHT, int(round(length / 0.005)), direction == 1, BLADE_HALF_WIDTH_CELLS)
        return bool(cells[row, col])

    def _push_blocker(self, cfg):
        """Nothing misplaced is visible: push the front-most object sideways towards free space (a feasible push)."""
        env, obs = self.env, self.env.obs
        ws = env.ws
        ys, xs = np.nonzero(obs['classmap'])
        if ys.size == 0:
            return None
        for _ in range(60):
            k = np.random.randint(ys.size)
            row, col = int(ys[k]), int(xs[k])
            direction = np.random.randint(2)
            index = direction * len(cfg.push_lengths) + len(cfg.push_lengths) // 2
            angle, length = decode_push(index, cfg.push_lengths)
            start_col = max(col - 4, 0)
            if check_push(ws, obs['depthmap'], start_col, row, angle, length):
                return 'push', (index, row, start_col)
        return None


def record_episodes(env, episodes, num_objects, seed=0, max_steps=40, log=print):
    """Run the expert; returns (list of transitions as tuples, statistics). Transitions are
    (state, action, next_state, reward, primitive, done) like the training loop's, with the hindsight pair added."""
    from l2r.learning.memory import Transition
    rng = np.random.default_rng(seed)
    transitions, solved, steps_total = [], 0, 0
    for episode in range(episodes):
        count = int(rng.integers(num_objects[0], num_objects[1] + 1))
        obs = env.reset(num_obj=count, seed=seed * 100003 + episode)
        expert, steps, success = ScriptedExpert(env), 0, False
        while steps < max_steps:
            choice = expert.act()
            if choice is None:
                break
            primitive, pix = choice
            full, grasped = env.obs, expert.object_at(choice[1]) if choice[0] == 'grasp' else None
            next_obs, reward, terminated, truncated, info = env.step({'primitive': primitive, 'pix': pix})
            expert.report(primitive, pix, grasped, info['executed'])
            transitions.append(Transition(obs, pix, next_obs, reward, primitive, bool(terminated)))
            if info['relabel']:
                obs_r, next_r = env.relabel(full, env.obs)
                transitions.append(Transition(obs_r, pix, next_r, env.cfg.reward_goal, primitive, True))
            obs = next_obs
            if info.get('phase') != 'place':
                steps += 1
            if terminated or truncated:
                success = bool(info['sorted'])
                break
        solved += success
        steps_total += steps
        log('episode %d (%d objects): %s in %d actions' % (episode, count, 'sorted' if success else 'not sorted', steps))
    stats = {'episodes': episodes, 'success_rate': solved / episodes, 'mean_actions': steps_total / episodes,
             'transitions': len(transitions)}
    return transitions, stats


def save_demos(path, transitions, stats):
    with open(path, 'wb') as f:
        pickle.dump({'transitions': [tuple(t) for t in transitions], 'stats': stats}, f, protocol=4)


def load_demos(path):
    from l2r.learning.memory import Transition
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return [Transition(*t) for t in data['transitions']], data['stats']
