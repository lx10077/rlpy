from collections import namedtuple
import math
import numpy as np
import random


Transition = namedtuple('Transition', ('state', 'action', 'mask', 'next_state', 'reward'))


# ====================================================================================== #
# Unlimited-length reply memory
# ====================================================================================== #
class NoLimitSequentialMemory(object):
    def __init__(self):
        self.memory = []

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(Transition(*args))

    def sample(self, batch_size=None):
        if batch_size is None:
            return Transition(*zip(*self.memory))
        else:
            random_batch = random.sample(self.memory, batch_size)
            return Transition(*zip(*random_batch))

    def append(self, new_memory):
        self.memory += new_memory.memory

    def __len__(self):
        return len(self.memory)


# ====================================================================================== #
# Limited-length reply memory
# ====================================================================================== #
class RingBuffer(object):
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = [None for _ in range(maxlen)]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]

    def append(self, v):
        if self.length < self.maxlen:
            self.length += 1
        elif self.length == self.maxlen:
            self.start = (self.start + 1) % self.maxlen
        else:
            raise RuntimeError()
        self.data[(self.start + self.length - 1) % self.maxlen] = v

    def clear(self):
        self.start = 0
        self.length = 0
        self.data = [None for _ in range(self.maxlen)]


class ReplayMemory(object):
    def __init__(self, max_size, batch_size):
        self.memory = RingBuffer(max_size)
        self.max_size = max_size
        self.batch_size = batch_size

    def add(self, *transitions):
        self.memory.append(Transition(*transitions))

    def sample(self, indexs=None):
        if indexs:
            assert isinstance(indexs, int)
            indexs = np.random.permutation(self.max_size)[:indexs]
            random_batch = [self.sample_one(ind) for ind in indexs]
        else:
            random_batch = [self.sample_one() for _ in range(self.batch_size)]
        return Transition(*zip(*random_batch))

    def sample_one(self, index=None):
        if index:
            assert isinstance(index, int) and 0 <= index < self.max_size
            ind = index
        else:
            ind = np.random.randint(0, self.__len__())
        return self.memory[ind]

    def clear(self):
        self.memory.clear()

    def __len__(self):
        return len(self.memory)

    def is_full(self):
        return True if self.__len__() >= self.max_size else False


# ====================================================================================== #
# Priorized reply memory
# ====================================================================================== #
class SumSegmentTree(object):
    """Uses sum segment binary tree to handle prioritization. """
    def __init__(self, max_size):
        self.max_size = max_size
        self.tree_level = int(math.ceil(math.log(max_size + 1, 2)) + 1)
        self.tree_size = 2 ** self.tree_level - 1
        self.tree = [0 for _ in range(self.tree_size)]
        self.data = [None for _ in range(self.max_size)]
        self.size = 0
        self.cursor = 0
        self.init_index = 2 ** (self.tree_level - 1) - 1

    def add(self, value, content):
        index = self.cursor
        self.data[index] = content
        self.__setitem__(index, value)

        self.cursor = (self.cursor + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def __setitem__(self, index, value):
        assert 0 <= index < self.max_size
        tree_index = self.init_index + index
        self.tree[tree_index] = value
        tree_index //= 2
        while tree_index > 0:
            self.tree[tree_index] = self.tree[2 * tree_index + 1] + self.tree[2 * (tree_index + 1)]
            tree_index //= 2
        self.tree[0] = self.tree[1] + self.tree[2]

    def __getitem__(self, index):
        assert 0 <= index < self.max_size
        tree_index = self.init_index + index
        return self.tree[tree_index]

    def _find(self, value):
        tree_index = 1
        while tree_index < self.init_index:
            left_value = self.tree[2 * tree_index + 1]
            if value <= left_value:
                tree_index = 2 * tree_index + 1
            else:
                value -= left_value
                tree_index = 2 * (tree_index + 1)
        index = tree_index - self.init_index
        return self.data[index], self.tree[tree_index], index

    def find(self, value, norm=True):
        if norm:  # if value is a fraction, multiply the sum of the tree.
            value *= self.tree[0]
        return self._find(value)

    def __len__(self):
        return self.size

    def __str__(self):  # print the value tree
        print_string = ''
        for k in range(1, self.tree_level + 1):
            for j in range(2 ** (k - 1) - 1, 2 ** k - 1):
                print_string += (str(self.tree[j]) + ' ')
            print_string += '\n'
        return print_string


class PriorizedReplayMemory(object):
    def __init__(self, max_size, batch_size, alpha, beta):
        """Prioritized replay memory initialization.

        Parameters
        ----------
        max_size : int, sample size to be stored
        batch_size: int, sample size to be sampled
        alpha: float, exponentially determine how much prioritization
                P(i) = priority_i ** alpha / sum(priority ** alpha)
        beta: float, exponentially affect importance-sampling (IS) weights
                w_i = (1 / N / P(i)) ** beta
        """
        self.tree = SumSegmentTree(max_size)
        self.memory_size = max_size
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.max_priority = 1.

    def add(self, *transitions):
        self.tree.add(self.max_priority ** self.alpha, Transition(*transitions))

    def sample(self):
        if len(self.tree) < self.batch_size:
            return None, None, None

        batch, weights, indices = [], [], []
        for _ in range(self.batch_size):
            r = random.random()
            data, priority, index = self.tree.find(r)
            batch.append(data)
            weights.append((self.memory_size * priority) ** (-self.beta) if priority > 1e-16 else 0)
            indices.append(index)
            self.priority_update([index], [0])  # To avoid duplicating

        max_weight = max(weights)  # Normalize for stability
        for i in range(len(weights)):
            weights[i] /= max_weight

        return Transition(*zip(*batch)), weights, indices

    def priority_update(self, indices, priorities):
        # After sampling proportionally (i.e. call self.sample()), priorities must be reverted,
        # which could be conducted as self.priority_update(indices, priorities).
        for i, p in zip(indices, priorities):
            self.tree[i] = p ** self.alpha
            self.max_priority = max(self.max_priority, p)

    def reset_alpha(self, alpha):
        self.alpha, old_alpha = alpha, self.alpha
        priorities = [self.tree[i] ** -old_alpha for i in range(len(self.tree))]
        self.priority_update(range(len(self.tree)), priorities)
