"""
    Implementation of unlimited rollouts buffer,
    capable of sampling both transitions and trajectories (for recurrent models)
    Used in Proximal Policy Optimization (PPO) and some other algorithms
    Copyright: Pavel B. Chernov, pavel.b.chernov@gmail.com
    Date: Dec 2020 - October 2021
    License: MIT
"""
import math
from typing import Sized, Optional, List, Any, Literal, Tuple
from collections import namedtuple
import copy
import numpy as np
import torch as th
from torch.utils.data.sampler import SubsetRandomSampler, BatchSampler
from torch.nn.utils.rnn import pad_sequence


Rollout = namedtuple(
    'Rollout',
    ('states', 'internal_states', 'actions', 'log_probs', 'rewards', 'raw_returns', 'raw_advantages', 'score')
)
Rollout.__annotations__ = {'states': th.Tensor, 'internal_states': th.Tensor, 'actions': th.Tensor,
                           'log_probs': th.Tensor, 'rewards': th.Tensor, 'raw_returns': th.Tensor,
                           'raw_advantages': th.Tensor, 'score': float}


class RolloutsBuffer(Sized):
    """
    Unlimited rollouts buffer, capable of sampling both transitions and trajectories (for recurrent models)

    Notes
    -----
    This buffer does NOT compute returns and advantages. They are to be precomputed beforehand.
    But it can normalize returns and/or advantages, if you specified these arguments as True.

    There is a useful method `get_rollout`, which returns a rollout by index.
    The rollout you get contains tensors, which are views to the main storage.
    It means that you can easily UPDATE values of rollout, for example: `raw_advantages`,
    by writing new values directly into the tensor.
    This is useful if you want to recompute advantages between each epoch of training.
    """
    
    def __init__(self,
                 normalize_returns=False,
                 normalize_advantages=True):
        self.normalize_returns = normalize_returns
        self.normalize_advantages = normalize_advantages
        
        # Intermediate lists to store rollouts from worker processes
        self._states: List[th.Tensor] = []
        self._internal_states: List[Any] = []
        self._actions: List[th.Tensor] = []
        self._log_probs: List[th.Tensor] = []
        self._rewards: List[th.Tensor] = []
        self._raw_returns: List[th.Tensor] = []
        self._raw_advantages: List[th.Tensor] = []
        self._indices = [0]
        self._scores = []
        
        # Final tensors to store all rollouts as tensors
        self.states = th.Tensor()
        self.internal_states = th.Tensor()
        self.actions = th.Tensor()
        self.log_probs = th.Tensor()
        self.rewards = th.Tensor()
        self.raw_returns = th.Tensor()
        self.returns = th.Tensor()
        self.raw_advantages = th.Tensor()
        self.advantages = th.Tensor()
        self.indices = th.Tensor()
        self.scores = th.Tensor()
    
    def reset(self):
        self._clear_rollout_buffers()
        self.states = th.Tensor()
        self.internal_states = th.Tensor()
        self.actions = th.Tensor()
        self.log_probs = th.Tensor()
        self.rewards = th.Tensor()
        self.raw_returns = th.Tensor()
        self.returns = th.Tensor()
        self.raw_advantages = th.Tensor()
        self.advantages = th.Tensor()
        self.indices = th.Tensor()
        self.scores = th.Tensor()
    
    def __len__(self):
        return len(self.states)
    
    def add(self, rollout: Rollout):
        self._states.extend(copy.copy(rollout.states))
        self._internal_states.extend(copy.copy(rollout.internal_states))
        self._actions.extend(copy.copy(rollout.actions))
        self._log_probs.extend(copy.copy(rollout.log_probs))
        self._rewards.extend(copy.copy(rollout.rewards))
        self._raw_returns.extend(copy.copy(rollout.raw_returns))
        self._raw_advantages.extend(copy.copy(rollout.raw_advantages))
        self._indices.append(len(self._states))
        self._scores.append(rollout.score)
    
    def finalize_buffer(self):
        # Convert lists to tensors
        self.indices = th.tensor(self._indices)
        self.scores = th.tensor(self._scores)
        self.states = th.stack(self._states)
        self.internal_states = th.stack(self._internal_states)
        self.actions = th.tensor(self._actions)
        self.log_probs = th.tensor(self._log_probs)
        self.rewards = th.tensor(self._rewards)
        self.raw_returns = th.tensor(self._raw_returns)
        self.returns = self.raw_returns
        self.raw_advantages = th.tensor(self._raw_advantages)
        self.advantages = self.raw_advantages
        
        if self.normalize_returns:
            self.renormalize_returns()
        if self.normalize_advantages:
            self.renormalize_advantages()
        
        self._clear_rollout_buffers()
    
    def renormalize_returns(self):
        with th.no_grad():
            returns = self.raw_returns
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            self.returns[:] = returns
    
    def renormalize_advantages(self):
        with th.no_grad():
            advantages = self.raw_advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            self.advantages[:] = advantages
    
    def _clear_rollout_buffers(self):
        self._states.clear()
        self._internal_states.clear()
        self._actions.clear()
        self._log_probs.clear()
        self._rewards.clear()
        self._raw_returns.clear()
        self._raw_advantages.clear()
        self._indices.clear()
        self._indices.append(0)
        self._scores.clear()
    
    def sample(self, batch_size: int, recurrent: bool = False, recurrent_sequence_length: Optional[int] = None):
        if recurrent:
            return self._sample_sequences(batch_size, recurrent_sequence_length)
        return self._sample_transitions(batch_size)
    
    def _sample_transitions(self,
                            batch_size: int
                            ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        assert th.is_tensor(self.states) and (len(self.states) > 0)
        # For non-recurrent models - sample a batch of transitions of the shape:
        # (N, x), where N - number of samples in a batch, x - state/action/return/advantage
        random_indices = SubsetRandomSampler(range(len(self.states)))
        sampler = BatchSampler(random_indices, batch_size, drop_last=False)
        for transition_indexes in sampler:
            states = self.states[transition_indexes]
            internal_states = self.internal_states[transition_indexes]
            actions = self.actions[transition_indexes]
            log_probs = self.log_probs[transition_indexes]
            returns = self.returns[transition_indexes]
            advantages = self.advantages[transition_indexes]
            yield states, internal_states, actions, log_probs, returns, advantages
    
    def _sample_sequences(self,
                          batch_size: int,
                          recurrent_sequence_length: Optional[int] = None
                          ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        assert th.is_tensor(self.states) and (len(self.states) > 0)
        # For recurrent models - sample a batch of rollouts of the shape:
        # (T, N, x), where T - the maximal length of a rollout,
        # N - number of rollouts in a batch, x - state/action/return/advantage
        if isinstance(recurrent_sequence_length, int) and (recurrent_sequence_length > 0):
            sequence_indices = []
            for i in range(len(self.indices) - 1):
                st, en = self.indices[i], self.indices[i + 1]
                offset = np.random.randint(recurrent_sequence_length)
                sequence_indices.extend(list(range(st + offset, en, recurrent_sequence_length)))
                if en - sequence_indices[-1] < recurrent_sequence_length:
                    sequence_indices[-1] = int(en) - recurrent_sequence_length
        else:
            sequence_indices = self.indices
        random_indices = SubsetRandomSampler(range(len(sequence_indices) - 1))
        sampler = BatchSampler(random_indices, batch_size, drop_last=True)
        for rollout_indexes in sampler:
            # indices has a batch_size number of indexes of rollouts
            states: List[th.Tensor] = []
            internal_states: List[th.Tensor] = []
            actions: List[th.Tensor] = []
            log_probs: List[th.Tensor] = []
            returns: List[th.Tensor] = []
            advantages: List[th.Tensor] = []
            for i in rollout_indexes:
                st, en = sequence_indices[i], sequence_indices[i + 1]
                if isinstance(recurrent_sequence_length, int) and (recurrent_sequence_length > 0):
                    en = st + recurrent_sequence_length
                states.append(self.states[st:en])
                # Note: for recurrent mode we use only starting internal state!
                # as model will further update its state on each step of a sequence
                internal_states.append(self.internal_states[st])
                actions.append(self.actions[st:en])
                log_probs.append(self.log_probs[st:en])
                returns.append(self.returns[st:en])
                advantages.append(self.advantages[st:en])
            # Convert lists to tensors padded up to the maximal sequence length
            states: th.Tensor = pad_sequence(states, batch_first=False)
            internal_states: th.Tensor = th.stack(internal_states)
            actions: th.Tensor = pad_sequence(actions, batch_first=False)
            log_probs: th.Tensor = pad_sequence(log_probs, batch_first=False)
            returns: th.Tensor = pad_sequence(returns, batch_first=False)
            advantages: th.Tensor = pad_sequence(advantages, batch_first=False)
            yield states, internal_states, actions, log_probs, returns, advantages
    
    @property
    def rollouts_count(self) -> int:
        return len(self.scores)
    
    @property
    def mean_rollout_length(self) -> Optional[int]:
        if len(self.indices) <= 0:
            return None
        return (self.indices[1:] - self.indices[:-1]).float().mean().item()
    
    def get_rollout(self, i: int) -> Rollout:
        st, en = self.indices[i], self.indices[i + 1]
        return Rollout(
            states=self.states[st:en],
            internal_states=self.internal_states[st:en],
            actions=self.actions[st:en],
            log_probs=self.log_probs[st:en],
            rewards=self.rewards[st:en],
            raw_returns=self.raw_returns[st:en],
            raw_advantages=self.raw_advantages[st:en],
            score=float(self.scores[i])
        )
    
    def export(self, n: int, method: Literal['best', 'worse', 'extreme', 'random'] = 'extreme') -> List[Rollout]:
        assert n <= len(self.scores), 'Requested number of episodes is more than available!'
        # Get rollout indices for export
        _, sorted_indices = self.scores.sort()
        if method == 'best':
            export_indices = sorted_indices[-n:].tolist()
        elif method == 'worse':
            export_indices = sorted_indices[:n].tolist()
        elif method == 'extreme':
            n_worse, n_best = math.ceil(n / 2), math.floor(n / 2)
            export_indices = sorted_indices[:n_worse].tolist() + sorted_indices[-n_best:].tolist()
        elif method == 'random':
            export_indices = np.random.choice(len(self.scores), n, replace=False).tolist()
        else:
            raise ValueError(f'Invalid {method=}')
        export_indices = set(export_indices)
        
        # Construct resulting list
        result = [self.get_rollout(i) for i in export_indices]
        return result
