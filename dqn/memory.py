import tensorflow as tf
import numpy as np

from functools import cached_property
from typing import Any, Callable

import random
import os
import shutil
import mmap
import sys

from math import ceil


class DataConverter:
    def serialize(self, item) -> bytes:
        raise NotImplementedError()

    def deserialize(self, raw) -> Any:
        raise NotImplementedError()

    @property
    def item_size(self) -> int:
        raise NotImplementedError()


class DQNConverter(DataConverter):
    def __init__(self, height: int, width: int, agent_history_length: int):
        super().__init__()
        self.height = height
        self.width = width
        self.agent_history_length = agent_history_length

    @cached_property
    def item_size(self) -> int:
        seq_size = self.height * self.width * self.agent_history_length * 4
        action_size = 8
        reward_size = 4
        next_seq_size = self.height * self.width * self.agent_history_length * 4
        bool_size = 1
        return (seq_size + action_size + reward_size + next_seq_size +
                bool_size)

    def serialize(self, item):
        assert len(item) == 5, 'invalid item'
        seq, action, reward, next_seq, done = item
        seq_raw = np.array(seq, dtype=np.float32).tobytes()
        action_raw = np.array(action, dtype=np.int64).tobytes()
        reward_raw = np.array(reward, dtype=np.float32).tobytes()
        next_seq_raw = np.array(next_seq, dtype=np.float32).tobytes()
        done_raw = np.array(done, dtype=np.bool).tobytes()
        return b''.join(
            [seq_raw, action_raw, reward_raw, next_seq_raw, done_raw])

    def deserialize(self, raw):
        assert len(raw) == self.item_size, 'invalid raw data'
        seq_size = self.height * self.width * self.agent_history_length * 4

        start, end = 0, seq_size
        seq_raw = raw[start:end]

        start, end = end, (end + 8)
        action_raw = raw[start:end]

        start, end = end, (end + 4)
        reward_raw = raw[start:end]

        start, end = end, (end + seq_size)
        next_seq_raw = raw[start:end]

        start, end = end, (end + 1)
        done_raw = raw[start:end]

        return (
            np.frombuffer(seq_raw, dtype=np.float32).reshape(
                (self.height, self.width, self.agent_history_length)),
            np.frombuffer(action_raw, dtype=np.int64).reshape(()),
            np.frombuffer(reward_raw, dtype=np.float32).reshape(()),
            np.frombuffer(next_seq_raw, dtype=np.float32).reshape(
                (self.height, self.width, self.agent_history_length)),
            np.frombuffer(done_raw, dtype=np.bool).reshape(()),
        )


class ReplayBuffer:
    def __init__(self, path: str, capacity: int, converter: DataConverter):
        self.path = path
        self.capacity = capacity
        self.converter = converter
        self.file = None
        self.data = None
        self.index = None
        self.size = None

    @cached_property
    def item_size(self):
        return self.converter.item_size

    @cached_property
    def raw_size(self):
        return 12 + self.capacity * self.item_size

    def initialize(self):
        total, used, free = shutil.disk_usage(os.path.dirname(self.path))
        assert free >= self.raw_size, 'not enough disk space'

        if sys.platform.startswith('linux'):
            r = os.system(f'fallocate -l {self.raw_size} {self.path}')
        elif sys.platform == 'darwin':
            r = os.system(f'mkfile -n {self.raw_size} {self.path}')
        else:
            raise Exception('Unexpected platform')

        assert r == 0, 'initialization failed'

        with open(self.path, 'r+b') as f:
            fn = f.fileno()
            mm = mmap.mmap(fn, 0)
            mm[:4] = np.array(self.capacity, dtype=np.int32).tobytes()
            mm[4:8] = np.array(0, dtype=np.int32).tobytes()
            mm[8:12] = np.array(0, dtype=np.int32).tobytes()
            mm.close()

    def open(self):
        try:
            self.file = open(self.path, 'r+b')
            self.data = mmap.mmap(self.file.fileno(), 0)
            self.data.madvise(mmap.MADV_RANDOM)
            assert np.frombuffer(
                self.data[:4],
                dtype=np.int32) == self.capacity, ('capacity does not match')

            self.index = np.frombuffer(self.data[4:8], dtype=np.int32).reshape(
                ())
            assert self.index >= 0 and self.index < self.capacity, (
                'index is out of bounds')

            self.size = np.frombuffer(self.data[8:12], dtype=np.int32).reshape(
                ())
            assert self.size >= 0 and self.size <= self.capacity, (
                'current size is out of bounds')
        except Exception:
            self.close()
            raise

    def close(self):
        if self.data is not None:
            self.data.close()
        if self.file is not None:
            self.file.close()
        self.index = None
        self.size = None

    def _write_to_index(self, x, index):
        start_index = 12 + self.item_size * index
        end_index = start_index + self.item_size
        self.data[start_index:end_index] = self.converter.serialize(x)

    def _read_from_index(self, index):
        start_index = 12 + self.item_size * index
        end_index = start_index + self.item_size
        return self.converter.deserialize(self.data[start_index:end_index])

    def _update_index(self, index):
        self.data[4:8] = np.array(index, dtype=np.int32).tobytes()
        self.index = index

    def _update_size(self, size):
        self.data[8:12] = np.array(size, dtype=np.int32).tobytes()
        self.size = size

    def append(self, x):
        assert self.capacity > 0
        self._write_to_index(x, self.index)
        self._update_index((self.index + 1) % self.capacity)
        if self.size < self.capacity:
            self._update_size(self.size + 1)

    def sample(self):
        buffer_size = int(self.size)
        return self._read_from_index(np.random.choice(buffer_size, (), False))

    def sample_batch(self, batch_size):
        samples = [
            self._read_from_index(x) for x in np.random.choice(
                self.size, min(self.size, batch_size), False)
        ]
        return tuple(tf.convert_to_tensor(np.stack(x)) for x in zip(*samples))
