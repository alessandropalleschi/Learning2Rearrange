"""n-step returns for the replay buffer (`--n_step N`): each stored transition carries the discounted sum of up to N real rewards and
the discount of its bootstrap, so the TD target r_0 + g_0 r_1 + ... + (g_0 ... g_{k-1}) Q(s_k) leans less on the (possibly optimistic)
value estimate of the next state and passes credit back N steps per update.

Per-step discount g_t follows the trainers: 1 for a grasp whose next state holds the object (pick and place are one decision),
gamma otherwise. The emitted transition's state carries '_nstep_discount' (the product of the g's over its steps) and '_outcome' (the
1-step outcome record of its first action, for the outcome heads)."""
from collections import deque

from l2r.learning.memory import Transition


def step_discount(transition, gamma):
    next_state, state = transition.next_state, transition.state
    holds_next = isinstance(next_state, dict) and next_state.get('held_class')
    holds_now = isinstance(state, dict) and state.get('held_class')
    return 1.0 if (holds_next and not holds_now) else gamma


class NStepAccumulator:
    def __init__(self, n, gamma):
        self.n, self.gamma = n, gamma
        self.queues = {}

    def _emit(self, steps):
        """Aggregate a run of consecutive (transition, discount) steps into one transition starting at the first."""
        first = steps[0][0]
        reward, discount = 0.0, 1.0
        for transition, g in steps:
            reward += discount * float(transition.reward)
            discount *= g
        last = steps[-1][0]
        state = dict(first.state)
        state['_nstep_discount'] = discount
        if isinstance(first.next_state, dict) and 'last_action' in first.next_state:
            state['_outcome'] = first.next_state['last_action']
        return Transition(state, first.action, last.next_state, reward, first.primitive_action, bool(last.done))

    def add(self, env, transition):
        """Add one environment step; returns the n-step transitions that are complete now."""
        queue = self.queues.setdefault(env, deque())
        queue.append((transition, step_discount(transition, self.gamma)))
        out = []
        if transition.done:  # the episode ended: every pending start gets its (shorter) return to the end
            while queue:
                out.append(self._emit(list(queue)))
                queue.popleft()
        elif len(queue) == self.n:
            out.append(self._emit(list(queue)))
            queue.popleft()
        return out

    def flush(self, env):
        """The episode was cut (truncated, or the simulator is reset): the pending starts bootstrap from the last next state."""
        queue = self.queues.get(env, deque())
        out = []
        while queue:
            out.append(self._emit(list(queue)))
            queue.popleft()
        return out
