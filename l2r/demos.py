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
        self.push_failures = {}  # object id -> pushes that were not executed (refused or stopped): not retried after MAX_FAILURES
        self.pushed_object = None  # the object the last proposed push was meant to move
        self.pull_failures = {}  # object id -> pulls that were not executed
        self.pulls = {}  # object id -> pulls proposed (at most MAX_FAILURES per object: a pull that worked is not repeated forever)
        self.extra = {}  # parameters of the last proposed action beyond (primitive, pix), e.g. a pull's approach / heading / length
        self.cleared = {}  # object id -> times moved out of another grasp's way (ordering rule)
        self.ordering = bool(getattr(env.cfg, 'expert_ordering', False))
        self.pushes = {}  # object id -> pushes made on it

    def object_at(self, pix):
        return int(self.env.obs['idmap'][pix[1], pix[2]])

    def report(self, primitive, pix, object_id, executed):
        if primitive == 'grasp' and not executed:
            self.failures[object_id] = self.failures.get(object_id, 0) + 1
        if primitive == 'pull' and not executed and object_id:
            self.pull_failures[object_id] = self.pull_failures.get(object_id, 0) + 1
        if primitive == 'push' and not executed:
            object_id = self.pushed_object if self.pushed_object is not None else object_id  # the push cell is beside it
            if object_id:
                self.push_failures[object_id] = self.push_failures.get(object_id, 0) + 1

    def _may_push(self, object_id):
        return self.push_failures.get(object_id, 0) < self.MAX_FAILURES

    def _misplaced_visible(self):
        env, scene = self.env, self.env.sim.scene
        idmap, found = env.obs['idmap'], []
        for object_id, (pos, _) in scene.poses().items():
            if scene.in_target(pos[1], scene.names[object_id], pos[0]):
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
        if choice is None or choice[0] != 'push' or not self.continuous or len(choice[1]) >= 5:
            return choice
        index, row, col = choice[1]
        lengths, (low, high) = self.cfg.push_lengths, self.env.cfg.push_length_range
        fraction = (lengths[index % len(lengths)] - low) / (high - low)
        return 'push', (index // len(lengths), row, col, float(np.clip(fraction, 0.0, 1.0)))

    PULL_APPROACH = np.radians(30)  # pull: in from the cabinet's middle towards the wall at this angle
    PULL_LENGTH = 0.08  # m, dragged away from the wall and towards the front (45 degrees)
    WALL_ZONE = 0.07  # m: an object whose centre is this close to a side wall counts as against it

    def _pull_from_wall(self, misplaced):
        """A pull for a misplaced object against a side wall that the grasps could not take (or None): attach from the
        cabinet's middle side and drag it away from the wall, so that it can be grasped next."""
        scene, b = self.env.sim.scene, self.env.ws.bounds
        poses = scene.poses()
        for _, object_id, row, col in misplaced:
            y = poses[object_id][0][1]
            near_wall = min(y - b[1][0], b[1][1] - y) < self.WALL_ZONE
            if (near_wall and self.failures.get(object_id, 0) >= self.MAX_FAILURES
                    and self.pulls.get(object_id, 0) < self.MAX_FAILURES):
                self.pulls[object_id] = self.pulls.get(object_id, 0) + 1
                self.failures[object_id] = 0  # off the wall it may be graspable again
                side = np.sign(y)  # +1: against the +y wall
                self.extra = {'approach': float(-side * self.PULL_APPROACH), 'heading': float(np.arctan2(-side, 1.0)),
                              'length': self.PULL_LENGTH}
                return 'pull', (0, row, col)
        return None

    def _act(self):
        self.pushed_object = None
        self.extra = {}
        env, cfg = self.env, self.cfg
        if env.holding:
            col, row = env.obs['heuristic_place']
            return 'place', (0, int(row), int(col))
        misplaced = self._misplaced_visible()
        candidates = [c for c in misplaced if self.failures.get(c[1], 0) < self.MAX_FAILURES]
        if candidates and self.ordering:
            choice = self._ordered_choice(candidates)
            if choice is not None:
                return choice
        if candidates:
            _, object_id, row, col = candidates[0]
            if self._near_region(object_id) and self._may_push(object_id):  # one push finishes it: no pick and place
                action = self._push_towards_target(object_id, cfg)
                if action:
                    self.pushes[object_id] = self.pushes.get(object_id, 0) + 1
                    self.pushed_object = object_id
                    return action
            return 'grasp', (0, row, col)
        pull = self._pull_from_wall(misplaced)  # an object stuck against a side wall: drag it off the wall first
        if pull is not None:
            return pull
        # nothing can be grasped: push a misplaced object (the one that resisted most) towards its target region
        for _, object_id, _, _ in sorted(misplaced, key=lambda c: -self.failures.get(c[1], 0)):
            if not self._may_push(object_id):
                continue
            action = self._push_towards_target(object_id, cfg)
            if action:
                self.pushed_object = object_id
                return action
        # every misplaced object is hidden behind others: push the front-most object aside to expose them
        action = self._expose_hidden(cfg)
        return action or self._push_blocker(cfg)

    BLOCKED_CELLS = 30  # raised cells of other objects in a grasp corridor above which the target counts as blocked

    def _corridor(self, row, col):
        """(obstacle cells, approach angle) of the grasp at (row, col) with the approach the environment will use."""
        from l2r.envs.approach import choose_approach, obstacles
        if getattr(self.env.cfg, 'grasp_approach', 'straight') == 'clear':
            approach, count = choose_approach(self.env.obs, (row, col))
            return count, approach
        return obstacles(self.env.obs, (row, col), 0.0), 0.0

    def _ordered_choice(self, candidates):
        """Ordering rule: the least blocked misplaced object first (front-most on ties); if even that one is blocked, first move the
        object that blocks its corridor most (at most once per object), so that the grasp does not knock it over."""
        from l2r.envs.approach import corridor_mask
        scored = sorted(((self._corridor(row, col)[0], -x, object_id, row, col) for x, object_id, row, col in candidates))
        cost, _, object_id, row, col = scored[0]
        if self._near_region(object_id):
            action = self._push_towards_target(object_id, self.cfg)
            if action:
                self.pushes[object_id] = self.pushes.get(object_id, 0) + 1
                return action
        if cost > self.BLOCKED_CELLS:
            _, approach = self._corridor(row, col)
            idmap = self.env.obs['idmap']
            mask = corridor_mask(idmap.shape, (row, col), approach, 8)
            ids, counts = np.unique(idmap[mask & (idmap != 0) & (idmap != object_id)], return_counts=True)
            for blocker in ids[np.argsort(-counts)]:
                blocker = int(blocker)
                if self.cleared.get(blocker, 0) >= 1 or self.failures.get(blocker, 0) >= self.MAX_FAILURES:
                    continue
                cells = np.argwhere(idmap == blocker)
                if len(cells) < 6:
                    continue
                self.cleared[blocker] = self.cleared.get(blocker, 0) + 1
                b_row, b_col = self._aim_cell(cells[:, 0], cells[:, 1])
                return 'grasp', (0, b_row, b_col)  # pick and place it (its goal, away from the misplaced objects' rows)
        return 'grasp', (0, row, col)

    def _near_region(self, object_id):
        """True if the object lies outside its target region but within one push (NEAR_REGION m) of it, and was not pushed
        before (a push that did not work is followed by a grasp)."""
        scene = self.env.sim.scene
        pos = scene.poses()[object_id][0]
        if hasattr(scene, 'goal_distance'):
            distance = scene.goal_distance(pos, scene.class_name(object_id))
        else:  # a scene without goal geometry (bands only)
            low, high = scene.regions[scene.class_name(object_id)]
            distance = max(low - pos[1], pos[1] - high, 0.0)
        transfer = 0.95 if getattr(getattr(self.env.sim, 'params', None), 'pusher', 'tip') == 'paddle' else 0.5  # measured push transfer
        reach = transfer * max(max(self.cfg.push_lengths) - 0.015, 0.0)  # how far one (longest) push really moves an object
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
        """Blade next to the object on the side away from its target region, moving towards the region. With goal slots
        ('grid' layout): sideways when the object is level with its slot, towards the back (a heading push) when it stands right in
        front of it, otherwise no push (pick and place)."""
        env, obs, scene = self.env, self.env.obs, self.env.sim.scene
        rows, cols = np.nonzero(obs['idmap'] == object_id)
        if rows.size == 0:
            return None
        low, high = scene.regions[scene.class_name(object_id)]
        x, y = scene.poses()[object_id][0][:2]
        if getattr(scene, 'layout', 'bands') != 'bands':
            x0, x1, y0, y1 = scene.slots[scene.class_name(object_id)]
            if not x0 <= x <= x1:
                if y0 <= y <= y1 and x > x1:  # straight in front of its slot: push it towards the back
                    return self._push_back(rows, cols, x - (x0 + x1) / 2)
                return None
        direction = 1 if (low + high) / 2 > y else 0  # 1: towards +y, 0: towards -y
        transfer = 0.95 if getattr(getattr(env.sim, 'params', None), 'pusher', 'tip') == 'paddle' else 0.5  # measured (scripts/physics_sanity.py)
        index = direction * len(cfg.push_lengths) + self._length_index(y, low, high, direction, cfg.push_lengths, transfer)
        angle, length = decode_push(index, cfg.push_lengths)
        height = obs['classmap'].shape[0]
        for gap in (3, 4, 2, 5):
            row = int(np.clip(rows.min() - gap if direction == 1 else rows.max() + gap, 0, height - 1))
            col = int(np.median(cols))
            if check_push(env.ws, obs['depthmap'], col, row, angle, length) and self._blade_touches(obs, row, col, direction, length):
                return 'push', (index, row, col)
        return None

    def _push_back(self, rows, cols, distance):
        """Heading push towards the back (pi) from just in front of the object, long enough to move it `distance` m."""
        transfer = 0.95 if getattr(getattr(self.env.sim, 'params', None), 'pusher', 'tip') == 'paddle' else 0.5
        low, high = self.env.cfg.push_length_range
        length = float(np.clip(distance / transfer + 0.015, low, high))
        row, col = int(np.median(rows)), int(min(cols.max() + 3, self.env.obs['idmap'].shape[1] - 1))
        return 'push', (0, row, col, (length - low) / (high - low), float(np.pi))

    @staticmethod
    def _length_index(y, low, high, direction, lengths, transfer=None):
        """Index of the push length that moves an object at `y` closest to the middle of its region [low, high]. A push moves it by
        `transfer` x (length - the 1.5 cm gap to the object) (measured: 0.5 with the tool tip, 0.95 with the paddle), or, without a
        transfer, by about its length (the original rule). With a single length that is index 0 (the paper's 8 cm)."""
        sign = 1.0 if direction == 1 else -1.0
        moved = [length if transfer is None else transfer * max(length - 0.015, 0.0) for length in lengths]
        landing = [abs((y + sign * m) - (low + high) / 2) for m in moved]
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


def make_expert(env):
    """The scripted expert, or the planning expert (`cfg.expert_planner` = number of candidates tried per decision)."""
    k = int(getattr(env.cfg, 'expert_planner', 0) or 0)
    if k > 0:
        from l2r.planner import PlanningExpert
        return PlanningExpert(env, candidates=k, horizon=int(getattr(env.cfg, 'planner_horizon', 3)))
    return ScriptedExpert(env)


def record_episodes(env, episodes, num_objects, seed=0, max_steps=40, log=print):
    """Run the expert; returns (list of transitions as tuples, statistics). Transitions are
    (state, action, next_state, reward, primitive, done) like the training loop's, with the hindsight pair added."""
    from l2r.learning.memory import Transition
    rng = np.random.default_rng(seed)
    transitions, solved, steps_total = [], 0, 0
    for episode in range(episodes):
        count = int(rng.integers(num_objects[0], num_objects[1] + 1))
        obs = env.reset(num_obj=count, seed=seed * 100003 + episode)
        expert, steps, success = make_expert(env), 0, False
        while steps < max_steps:
            choice = expert.act()
            if choice is None:
                break
            primitive, pix = choice
            full, grasped = env.obs, expert.object_at(choice[1]) if choice[0] in ('grasp', 'push', 'pull') else None
            next_obs, reward, terminated, truncated, info = env.step({'primitive': primitive, 'pix': pix, **expert.extra})
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


def adapt_demos(transitions, cfg):
    """Make demonstrations recorded with a continuous push length (pix = (direction, row, col, fraction)) usable by an agent with
    selectable lengths (`cfg.push_lengths`): the length becomes the nearest selectable one (pix = (direction * L + index, row, col)).
    Demonstrations with selectable lengths are returned unchanged; they cannot be used with a continuous length (unknown lengths).
    Without a belief (`cfg.belief == 'none'`) the recorded spatial belief is removed from the observations: the agent must not see a
    memory its environment does not give it (the object agent turns any belief it finds into ghost tokens)."""
    from l2r.learning.memory import Transition
    if getattr(cfg, 'skill_params', False):  # parameterised skills: headings / approach angles in the action tuple
        from l2r.learning.skill_params import adapt_demo_actions
        transitions = adapt_demo_actions(transitions)
    lengths, (low, high) = np.asarray(cfg.push_lengths), cfg.push_length_range
    out = []
    strip = cfg.belief == 'none'
    for t in transitions:
        if strip and ('belief' in t.state or 'belief' in t.next_state):
            t = Transition({k: v for k, v in t.state.items() if k != 'belief'}, t.action,
                           {k: v for k, v in t.next_state.items() if k != 'belief'}, *t[3:])
        if t.primitive_action == 'push' and len(t.action) == 4 and not cfg.hierarchy:
            direction, row, col, fraction = t.action
            index = int(np.argmin(np.abs(lengths - (low + fraction * (high - low)))))
            t = Transition(t.state, (int(direction) * len(lengths) + index, int(row), int(col)), *t[2:])
        elif t.primitive_action == 'push' and len(t.action) == 3 and cfg.hierarchy:
            raise ValueError('demonstrations with selectable push lengths cannot train a continuous push length: record them with '
                             '--hierarchy')
        out.append(t)
    return out


def load_demos(path):
    from l2r.learning.memory import Transition
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return [Transition(*t) for t in data['transitions']], data['stats']
