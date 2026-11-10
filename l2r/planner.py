"""Planning expert (`--expert_planner`): the scripted expert's candidate actions, each tried in the simulator from a saved state
(the action, the placement that follows a grasp, then a short rollout of the scripted expert), scored by what it sorted, against
toppling and the actions spent; the best is executed. Search with real outcomes instead of rules: it finds clearing moves and
orderings the rules miss. Slower per decision (candidates x horizon simulated actions); meant for recording demonstrations.

Saved state: PyBullet's saveState (bodies, joints, velocities) plus the environment's and the expert's own fields; the planner only
plans when nothing is held (no suction constraint lives across a snapshot)."""
import copy

import numpy as np

from l2r.demos import ScriptedExpert

ENV_FIELDS = ('actions', 'episode_return', 'holding', '_phi', 'obs', '_last_action', '_last_pix', '_relabel_state', '_positions')
EXPERT_FIELDS = ('failures', 'pushes', 'cleared')


class PlanningExpert(ScriptedExpert):
    def __init__(self, env, candidates=4, horizon=3, push_lengths=None):
        super().__init__(env, push_lengths)
        self.n_candidates, self.horizon = candidates, horizon

    # ------------------------------------------------------------------ state
    def _snapshot(self):
        env = self.env
        return (env.sim.p.saveState(), {k: copy.deepcopy(getattr(env, k)) for k in ENV_FIELDS if hasattr(env, k)},
                copy.deepcopy(env.tracker.state), {k: copy.deepcopy(getattr(self, k)) for k in EXPERT_FIELDS})

    def _restore(self, snapshot, remove=False):
        state_id, env_fields, tracker, expert_fields = snapshot
        env = self.env
        tool = env.sim.tool
        if getattr(tool, 'activated', False):
            tool.release()  # a grasp inside the trial: drop its constraint before restoring the bodies
        env.sim.p.restoreState(stateId=state_id)
        for k, v in env_fields.items():
            setattr(env, k, copy.deepcopy(v))
        env.tracker.state = copy.deepcopy(tracker)
        for k, v in expert_fields.items():
            setattr(self, k, copy.deepcopy(v))
        if remove:
            env.sim.p.removeState(state_id)

    # ------------------------------------------------------------------ candidates
    def _candidates(self):
        """The rule's own choice first, then grasps of the other graspable misplaced objects (front-most first) and pushes of
        the objects near their goal."""
        out = []
        base = ScriptedExpert.act(self)
        if base is not None:
            out.append(base)
        for _, object_id, row, col in self._misplaced_visible():
            if self.failures.get(object_id, 0) < self.MAX_FAILURES:
                out.append(('grasp', (0, row, col)))
            if self._near_region(object_id):
                push = self._push_towards_target(object_id, self.cfg)
                if push:
                    out.append(self._convert(push))
        unique = []
        for c in out:
            if all(c[0] != u[0] or tuple(c[1]) != tuple(u[1]) for u in unique):
                unique.append(c)
        return unique[:self.n_candidates]

    def _convert(self, choice):
        """A rule action as the environment needs it (continuous push lengths: the length as a fraction)."""
        if choice[0] != 'push' or not self.continuous or len(choice[1]) >= 5:
            return choice
        index, row, col = choice[1]
        lengths, (low, high) = self.cfg.push_lengths, self.env.cfg.push_length_range
        return 'push', (index // len(lengths), row, col, float(np.clip((lengths[index % len(lengths)] - low) / (high - low), 0, 1)))

    # ------------------------------------------------------------------ search
    def _execute(self, choice):
        """One decision: the action (and, for a grasp that holds the object, the expert's placement). Returns (done, invalid)."""
        env = self.env
        _, _, terminated, truncated, info = env.step({'primitive': choice[0], 'pix': choice[1]})
        if info.get('phase') == 'place' and not (terminated or truncated):
            col, row = env.obs['heuristic_place']
            _, _, terminated, truncated, info = env.step({'primitive': 'place', 'pix': (0, int(row), int(col))})
        return terminated or truncated, bool(info.get('invalid', False))

    def _score(self, choice):
        done, invalid = self._execute(choice)
        steps = 1
        while not done and steps <= self.horizon:
            nxt = ScriptedExpert.act(self)
            if nxt is None:
                break
            done, invalid = self._execute(nxt)
            steps += 1
        if invalid:
            return -10.0 - steps * 0.1
        return 10.0 * self.env.sim.scene.fraction_sorted() - steps * 0.1

    def act(self):
        if self.env.holding:  # a placement: the rule (the planner plans pick-and-place as one decision)
            return ScriptedExpert.act(self)
        counters = {k: copy.deepcopy(getattr(self, k)) for k in EXPERT_FIELDS}
        candidates = self._candidates()  # the rule may update its counters while proposing: undo that
        for k, v in counters.items():
            setattr(self, k, v)
        if len(candidates) <= 1:
            return candidates[0] if candidates else None
        snapshot = self._snapshot()
        scores = []
        for choice in candidates:
            scores.append(self._score(choice))
            self._restore(snapshot)
        self._restore(snapshot, remove=True)
        self.last_scores = list(zip(candidates, scores))
        return candidates[int(np.argmax(scores))]
