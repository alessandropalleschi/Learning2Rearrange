import random
from collections import namedtuple, deque

import numpy as np

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'primitive_action', 'done'))


class ReplayMemory(object):
    """Uniform replay buffer."""

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class PrioritizedReplay(object):
    """Proportional prioritized replay (Schaul et al. 2016) with importance-sampling weights.

    New transitions get the current maximum priority; `update` sets priority = (|TD error| + eps)^alpha.
    """

    def __init__(self, capacity, alpha=0.6, eps=1e-3):
        self.capacity, self.alpha, self.eps = capacity, alpha, eps
        self.data = []
        self.priorities = np.zeros(capacity, np.float64)
        self.pos = 0

    def push(self, *args):
        t = Transition(*args)
        max_p = self.priorities[:len(self.data)].max() if self.data else 1.0
        if len(self.data) < self.capacity:
            self.data.append(t)
        else:
            self.data[self.pos] = t
        self.priorities[self.pos] = max_p
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        """Returns (transitions, indices, importance weights normalised to max 1)."""
        p = self.priorities[:len(self.data)]
        probs = p / p.sum()
        idx = np.random.choice(len(self.data), batch_size, p=probs)
        weights = (len(self.data) * probs[idx]) ** (-beta)
        weights = (weights / weights.max()).astype(np.float32)
        return [self.data[i] for i in idx], idx, weights

    def update(self, idx, td_errors):
        self.priorities[idx] = (np.abs(td_errors) + self.eps) ** self.alpha

    def __len__(self):
        return len(self.data)
