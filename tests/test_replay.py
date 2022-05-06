import random
import unittest

import numpy as np

from optim.replay import ReplayBuffer, PrioritizedReplayBuffer


class TestReport(unittest.TestCase):
    def test_replay(self):
        buf = ReplayBuffer(5)
        for c in '0123456789':
            buf.put(c)
        self.assertTrue(len(buf), 5)
        for c in '01234':
            self.assertFalse(c in buf.buffer)
        for c in '56789':
            self.assertTrue(c in buf.buffer)
            
    def test_priority_alpha_0(self):
        buf = PrioritizedReplayBuffer(64, alpha=0.0)
        for _ in range(buf.capacity//2):
            buf.put('A', 0)
        for _ in range(buf.capacity//2):
            buf.put('B', 1)
        arr, w, idx = buf.sample(10000, beta=1.0)
        uniq, freq = np.unique(arr, return_counts=True)
        self.assertEqual(freq[0], 5000)
        self.assertEqual(freq[1], 5000)

    def test_priority_alpha_1(self):
        buf = PrioritizedReplayBuffer(256, alpha=1.0)
        for _ in range(buf.capacity//2):
            buf.put('A', 0)
        for _ in range(buf.capacity//2):
            buf.put('B', 1)
        arr, w, idx = buf.sample(512, beta=1.0)
        uniq, freq = np.unique(arr, return_counts=True)
        print(freq)
        self.assertEqual(freq[0], 5)
        self.assertEqual(freq[1], 507)

        buf = PrioritizedReplayBuffer(256, alpha=1.0)
        for _ in range(buf.capacity//2):
            buf.put('A', 1)
        for _ in range(buf.capacity//2):
            buf.put('B', 1)
        arr, w, idx = buf.sample(512, beta=1.0)
        uniq, freq = np.unique(arr, return_counts=True)
        print(freq)
        self.assertEqual(freq[0], 256)
        self.assertEqual(freq[1], 256)

    def test_priority_put_without_value(self):
        buf = PrioritizedReplayBuffer(512, alpha=0.6)
        for _ in range(buf.capacity//8):
            buf.put(random.choice('ABCDEF'))
        arr, w, idx = buf.sample(512, beta=0.0)
        self.assertTrue(all([x is not None for x in arr]))

    def test_priority_large_buffer(self):
        buf = PrioritizedReplayBuffer(20000, alpha=0.6)
        for _ in range(5000):
            buf.put(random.choice('ABCDEF'))
        self.assertEqual(len(buf), 5000)
        arr, w, idx = buf.sample(64, beta=0.0)
        self.assertTrue(all([x is not None for x in arr]))


if __name__ == '__main__':
    unittest.main()
