from collections import deque
import random
import numpy as np


class MemoryBuffer:
    """
    # ---Memory Buffer---
    """

    def __init__(self, size):
        self.buffer = deque(maxlen=size)
        self.maxSize = size
        self.len = 0

    def sample(self, count):
        batch = []
        batch = random.sample(self.buffer, count)

        s_array = np.float32([array[0] for array in batch])
        a_array = np.float32([array[1] for array in batch])
        r_array = np.float32([array[2] for array in batch])
        new_s_array = np.float32([array[3] for array in batch])
        done_array = np.float32([array[4] for array in batch])

        return s_array, a_array, r_array, new_s_array, done_array

    def len(self):
        return len(self.buffer)

    def add(self, s, a, r, new_s, done):
        transition = (s, a, r, new_s, done)
        # self.len += 1
        # if self.len > self.maxSize:
        #     self.len = self.maxSize
        self.buffer.append(transition)
