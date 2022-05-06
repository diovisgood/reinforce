"""
    Implementation of different replay buffers for Reinforcement Learning algorithms:
    - ReplayBuffer - simple limited size buffer with uniform sampling
    - PrioritizedReplayBuffer - simple limited size buffer with prioritized sampling
    - RolloutBuffer
    Copyright: Pavel B. Chernov, pavel.b.chernov@gmail.com
    Date: Dec 2020 - July 2021
    License: MIT
"""
from typing import Any, Sized, List, Tuple, Optional, Sequence
from collections import deque
import numpy as np
import torch as th
import random


class ReplayBuffer(Sized):
    """Simple limited size buffer with uniform sampling"""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def append(self, record: Any):
        self.buffer.append(record)

    def add(self, record: Any):
        self.buffer.append(record)

    def put(self, record: Any):
        self.buffer.append(record)

    def __len__(self):
        return len(self.buffer)
    
    def sample(self, n: int) -> List:
        return random.sample(self.buffer, n)


class PrioritizedReplayBuffer(Sized):
    """
    Limited size buffer which samples records according to their weights (priorities)

    Based on SumTree - a binary tree data structure which can store some values associated with its leafs,
    and where the parent’s value is the sum of its children values.

    Parameters
    ----------
    capacity : int
        Specify the maximum size of the buffer.
    alpha : float
        Alpha value for prioritized sample selection in a range: [0, 1].
        This parameter determines how much prioritization is used.
        When alpha=0 - selection is done uniformly.
        When alpha=1 - only select records with the highest priority.
        Default: 0.6
    eps : float
        A small constant added to each priority value,
        to ensure that no record would has a zero probability to be selected.
        Default: 0.01
    """
    def __init__(self, capacity: int, alpha: float = 0.6, eps: float = 0.01):
        assert 0 <= alpha <= 1
        
        self.capacity = capacity
        self.alpha = alpha
        self.eps = eps

        # Ensure tree capacity is the power of two, otherwise we can't construct a binary tree
        tree_capacity = int(2 ** np.ceil(np.log2(capacity)))
        
        # Initialize tree storage, records storage and records counter
        self.tree = np.zeros(2 * tree_capacity, dtype=np.float32)
        self.data = []
        self.counter = 0

    def __len__(self):
        return len(self.data)
    
    @property
    def total_priority(self):
        """Get the sum of all priorities in the tree"""
        return self.tree[1]
    
    def append(self, record: Any, value: Optional[float] = None):
        """Insert new record into the tree along with its value"""
        self.put(record, value)

    def add(self, record: Any, value: Optional[float] = None):
        """Insert new record into the tree along with its value"""
        self.put(record, value)

    def put(self, record: Any, value: Optional[float] = None):
        """Insert new record into the tree along with its value"""
        # Compute priority
        if value is not None:
            priority = (abs(value) + self.eps) ** self.alpha
        else:
            priority = 1.0 ** self.alpha
        
        # Add record
        index = self.counter % self.capacity
        if index >= len(self.data):
            self.data.append(record)
        else:
            self.data[index] = record
        self.counter += 1
        
        # Set priority in the tree
        node = index + self.capacity
        # priority_change = priority - self.tree[node]
        self.tree[node] = priority
        
        # Update priority in the tree up to the root
        while node > 1:
            node = node // 2
            # self.tree[node] += priority_change
            self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def sample(self, n: int, beta: float = 1.0) -> Tuple[List[Any], np.ndarray, List[int]]:
        """
        Get a batch of samples from buffer
        
        Parameters
        ----------
        n : int
            The required number of samples
        beta : float
            This parameter determines to what degree to use importance weights
            (0 - no corrections, 1 - full correction).
            Typically you mau want to slowly increase beta from 0 to 1 during training,
            as model performance increases.

        Returns
        -------
        records : List[Any]
            List of sampled records.
        weights : np.array
            Array of shape (batch_size,) and dtype np.float32
            denoting importance weight of each record.
        indexes : List[int]
            Array of shape (batch_size,) and dtype np.int32
            indexes in buffer of sampled records
        """
        assert 0 <= beta <= 1
        records, priorities, indexes = [], [], []
        segment = self.total_priority / n

        for i in range(n):
            random_priority_in_segment = random.random() * segment + i * segment
            record, priority, index = self.get(random_priority_in_segment)
            records.append(record)
            priorities.append(priority)
            indexes.append(index)
            
        priorities = np.asarray(priorities, dtype=np.float32)
        probabilities: np.ndarray = priorities / self.total_priority
        weights: np.ndarray = np.power(len(self.data) * probabilities, -beta)
        weights /= np.max(weights)
        
        return records, weights, indexes

    def update(self, indexes: Sequence[int], values: Sequence[float]):
        """Update values for the records, specified by indexes"""
        assert len(indexes) == len(values)
        for index, value in zip(indexes, values):
            assert 0 <= index < len(self.data)
            # Set priority in the tree
            priority = (abs(value) + self.eps) ** self.alpha
            node = index + self.capacity
            priority_change = priority - self.tree[node]
            self.tree[node] = priority
            
            # Update priority in the tree up to the root
            while node > 1:
                node = node // 2
                self.tree[node] += priority_change

    def get(self, target_priority: float) -> Tuple[Any, float, int]:
        """
        Find the highest index in the array such that
            sum(arr[0] + arr[1] + ... + arr[i - i]) <= target_priority
        If array values are probabilities, this function allows
        to sample indexes according to the discrete probability efficiently.
        """
        # Traverse the tree from top to bottom until leaf
        node = 1
        while node < self.capacity:
            left_child_priority = self.tree[2 * node]
            if left_child_priority > target_priority:
                # Choose left child
                node = 2 * node
            else:
                # Choose right child
                target_priority -= left_child_priority
                node = 2 * node + 1
        index = node - self.capacity
        if (index < 0) or (index >= len(self.data)):
            # In case of invalid index - return first record
            import warnings
            warnings.warn(f'Invalid index {node=}, {index=}, len(data)={len(self.data)}')
            return self.data[0], self.tree[self.capacity], 0
        return self.data[index], self.tree[node], index


class RolloutBuffer(Sized):
    """
    Unlimited buffer to store rollouts
    
    Allows to sample either:
    - transitions: (state, action, advantage, reward)
    - trajectories: a sequence of transitions.
    
    Latter is useful to train recurrent models.
    
    Parameters
    ----------
    future_discount_factor : float
        Default: 0.99
    recurrent : bool
        Default: False
    """
    eps = 1e-4
    
    def __init__(self, capacity: int, recurrent=False):
        self.recurrent = recurrent
        self.states = []
        self.actions = []
        self.returns = []
        self.advantages = []
        self.size = 0
        self.index = [0]
        self.reset()
    
    def __len__(self):
        return len(self.states)
    
    def reset(self):
        self.states.clear()
        self.actions.clear()
        self.returns.clear()
        self.advantages.clear()
        self.size = 0
        self.index = [0]
    
    def push(self, state, action, value, reward, done=False):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.size += 1
        if done:
            self.index.append(self.size)
            rewards = self.rewards[self.index[-2]:self.index[-1]]
            returns = self._compute_returns(rewards, self.future_discount_factor)
            self.returns.extend(returns)

    @staticmethod
    def _compute_returns(rewards: List[th.Tensor], future_discount_factor: float) -> th.Tensor:
        returns = []
        R = 0
        for reward in reversed(rewards):
            R = reward + R * future_discount_factor
            returns.insert(0, R)
        returns = th.tensor(returns)
        return returns
    
    def _finish_buffer(self):
        with th.no_grad():
            self.states = th.tensor(self.states)
            self.actions = th.tensor(self.actions)
            self.rewards = th.tensor(self.rewards)
            self.returns = th.tensor(self.returns)
            self.values = th.tensor(self.values)
            advantages = self.returns - self.values
            advantages = (advantages - advantages.mean()) / (advantages.std() + self.eps)
            self.advantages = advantages
            self.values.clear()
            self.buffer_ready = True

    def sample(self, batch_size=64, recurrent=False):
        from torch.utils.data.sampler import SubsetRandomSampler, BatchSampler

        if not self.buffer_ready:
            self._finish_buffer()
        
        if recurrent:
            random_indices = SubsetRandomSampler(range(len(self.index) - 1))
            sampler = BatchSampler(random_indices, batch_size, drop_last=True)
            
            for traj_indices in sampler:
                states = [self.states[self.index[i]:self.index[i + 1]] for i in traj_indices]
                actions = [self.actions[self.index[i]:self.index[i + 1]] for i in traj_indices]
                returns = [self.returns[self.index[i]:self.index[i + 1]] for i in traj_indices]
                advantages = [self.advantages[self.index[i]:self.index[i + 1]] for i in traj_indices]
                traj_mask = [th.ones_like(r) for r in returns]
                
                states = pad_sequence(states, batch_first=False)
                actions = pad_sequence(actions, batch_first=False)
                returns = pad_sequence(returns, batch_first=False)
                advantages = pad_sequence(advantages, batch_first=False)
                traj_mask = pad_sequence(traj_mask, batch_first=False)
                
                yield states, actions, returns, advantages, traj_mask
        
        else:
            random_indices = SubsetRandomSampler(range(self.size))
            sampler = BatchSampler(random_indices, batch_size, drop_last=True)
            
            for i, idxs in enumerate(sampler):
                states = self.states[idxs]
                actions = self.actions[idxs]
                returns = self.returns[idxs]
                advantages = self.advantages[idxs]
                
                yield states, actions, returns, advantages, 1


def merge_buffers(buffers):
    memory = RolloutBuffer()
    
    for b in buffers:
        offset = len(memory)
        
        memory.states += b.states
        memory.actions += b.actor
        memory.rewards += b.rewards
        memory.values += b.values
        memory.returns += b.returns
        
        memory.ep_returns += b.ep_returns
        memory.ep_lens += b.ep_lens
        
        memory.index += [offset + i for i in b.index[1:]]
        memory.size += b.size
    return memory
