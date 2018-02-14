"""Memory class"""

from pickle import Pickler, Unpickler
from pathlib import Path

import random
import numpy as np

class Memory:
    """Circular memory buffer that is smart enough
    to return next_state without explicitly storing it"""
    def __init__(self, size, state_shape, action_shape):
        self.size = size
        self._index = 0
        self._count = 0
        self._last_insert = 0
        self.storage = [
            np.zeros((size,) + state_shape, dtype=np.float32),  # state
            np.zeros((size,) + action_shape, dtype=np.float32), # action
            np.zeros((size,), dtype=np.float32), # reward
            np.zeros((size,), dtype=np.float32), # done
        ]

    @staticmethod
    def fname(path):
        return path + ".memory"

    @classmethod
    def restore(self, path):
        with open(self.fname(path), 'rb') as f:
            p = Unpickler(f)
            return p.load()

    def save(self, path):
        with open(self.fname(path), 'wb') as f:
            p = Pickler(f, -1)
            p.dump(self)
        # remove old *.memory files
        save_dir = Path(path).parent
        for memfile in save_dir.glob("*.memory"):
            basefile = memfile.with_suffix('.meta')
            if not basefile.is_file():
                memfile.unlink()

    def count(self):
        return self._count

    def remember(self, state, action, reward, done):
        for col, val in zip(self.storage, [state, action, reward, done]):
            col[self._index] = val
        self._count += 1
        self._last_insert = self._index
        self._index = (self._index + 1) % self.size

    def sample(self, batch_size):
        """returns the next state for each sample as well

        So full return is [[states], [actions], [rewards], [dones], [next_states]]"""
        count = self._count
        if count <= batch_size:
            return
        if count > self.size:
            count = self.size
        # we can't use random.sample(range(count)) here because
        # there's an element in the middle of the range
        # (self._last_insert) that we need to reject.
        indices = []
        while len(indices) < batch_size:
            n = random.randrange(count)
            if n != self._last_insert and n not in indices:
                indices.append(n)
        indices = np.array(indices)
        samples = [col[indices] for col in self.storage]
        # add the next states as well
        next_indices = (indices + 1) % self.size
        samples.append(self.storage[0][next_indices])
        return samples