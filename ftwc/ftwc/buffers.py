import random

import numpy as np
# from collections import namedtuple
from rlpyt.utils.collections import namedarraytuple

# a snapshot of state to be stored in replay memory
Transition = namedarraytuple('Transition', ('obs_word_ids', 'cmd_word_ids',
                                       'reward', 'mask', 'done',
                                       'next_obs_word_ids',
                                       'next_word_masks'))


class HistoryScoreCache(object):

    def __init__(self, capacity=1):
        self.memory = []
        self.capacity = capacity
        self.reset()

    def push(self, stuff):
        """stuff is float."""
        if len(self.memory) < self.capacity:
            self.memory.append(stuff)
        else:
            self.memory = self.memory[1:] + [stuff]

    def get_avg(self):
        return np.mean(np.array(self.memory))

    def reset(self):
        self.memory = []

    def __len__(self):
        return len(self.memory)


class PrioritizedReplayMemory(object):

    def __init__(self, capacity=100000, priority_fraction=0.0):
        # prioritized replay memory
        self.priority_fraction = priority_fraction
        self.alpha_capacity = int(capacity * priority_fraction)
        self.beta_capacity = capacity - self.alpha_capacity
        self.alpha_memory, self.beta_memory = [], []
        self.alpha_position, self.beta_position = 0, 0

    def push(self, is_priority=False, transition: Transition = None):
        """Saves a transition."""
        assert transition is not None

        if self.priority_fraction == 0.0:
            is_priority = False
        if is_priority:
            if len(self.alpha_memory) < self.alpha_capacity:
                self.alpha_memory.append(None)
            self.alpha_memory[self.alpha_position] = transition #Transition(*args)
            self.alpha_position = (self.alpha_position + 1) % self.alpha_capacity
        else:
            if len(self.beta_memory) < self.beta_capacity:
                self.beta_memory.append(None)
            self.beta_memory[self.beta_position] = transition
            self.beta_position = (self.beta_position + 1) % self.beta_capacity

    def sample(self, batch_size):
        if self.priority_fraction == 0.0:
            from_beta = min(batch_size, len(self.beta_memory))
            res = random.sample(self.beta_memory, from_beta)
        else:
            from_alpha = min(int(self.priority_fraction * batch_size), len(self.alpha_memory))
            from_beta = min(batch_size - int(self.priority_fraction * batch_size), len(self.beta_memory))
            res = random.sample(self.alpha_memory, from_alpha) + random.sample(self.beta_memory, from_beta)
        random.shuffle(res)
        return res

    def __len__(self):
        return len(self.alpha_memory) + len(self.beta_memory)
