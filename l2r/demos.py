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

    def __init__(self, env):
        self.env = env
        self.failures = {}  # object id -> failed grasps

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
                front = cols.max()
                at_front = rows[cols >= front - 1]
                # aim INSIDE the object: the approach stops on contact, but the suction only engages on a real contact point and
                # the surface at cup height is often a few mm behind the front-most cell of the top view (tapered bottles)
                aim = max(front - GRASP_DEPTH_CELLS, int(cols.min()))
                found.append((pos[0], object_id, int(np.median(at_front)), int(aim)))
        return sorted(found, reverse=True)  # largest x first: nothing is in front of it

    def act(self):
        """(primitive, pix) or None if the expert has nothing sensible to do."""
        env, cfg = self.env, self.env.cfg
        if env.holding:
            col, row = env.obs['heuristic_place']
            return 'place', (0, int(row), int(col))
        misplaced = self._misplaced_visible()
        candidates = [c for c in misplaced if self.failures.get(c[1], 0) < self.MAX_FAILURES]
        if candidates:
            _, _, row, col = candidates[0]
            return 'grasp', (0, row, col)
        # nothing can be grasped: push a misplaced object (the one that resisted most) towards its target region
        for _, object_id, _, _ in sorted(misplaced, key=lambda c: -self.failures.get(c[1], 0)):
            action = self._push_towards_target(object_id, cfg)
            if action:
                return action
        return self._push_blocker(cfg)

    def _push_towards_target(self, object_id, cfg):
        """Blade next to the object on the side away from its target region, moving towards the region."""
        env, obs, scene = self.env, self.env.obs, self.env.sim.scene
        rows, cols = np.nonzero(obs['idmap'] == object_id)
        if rows.size == 0:
            return None
        low, high = scene.regions[scene.class_name(object_id)]
        y = scene.poses()[object_id][0][1]
        direction = 1 if (low + high) / 2 > y else 0  # 1: towards +y, 0: towards -y
        index = direction * len(cfg.push_lengths) + len(cfg.push_lengths) // 2
        angle, length = decode_push(index, cfg.push_lengths)
        height = obs['classmap'].shape[0]
        for gap in (3, 4, 2, 5):
            row = int(np.clip(rows.min() - gap if direction == 1 else rows.max() + gap, 0, height - 1))
            col = int(np.median(cols))
            if check_push(env.ws, obs['depthmap'], col, row, angle, length):
                return 'push', (index, row, col)
        return None

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
