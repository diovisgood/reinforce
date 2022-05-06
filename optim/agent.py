from typing import Union, Tuple, Type, Optional, Mapping, Any
import gym.spaces as spaces
import torch as th
import numpy as np
import torch.nn as nn
# import torch.nn.functional as F
from torch.distributions import Distribution, Categorical
from abc import ABC, abstractmethod
# from multiprocessing.context import BaseContext

from optim.models import BaseModel, MLP
from modules.standardize import Standardize


class Agent(nn.Module, ABC):
    def __init__(self,
                 observation_space: spaces.Space,
                 action_space: spaces.Space,
                 normalize_input: Union[bool, th.BoolTensor] = True,
                 # mp_context: Optional[BaseContext] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.observation_space = observation_space
        self.action_space = action_space
        self.input_size = space_size(observation_space)
        self.output_size = space_size(action_space)
        self.normalize_input = False
        self.norm: Optional[nn.Module] = None
        if (isinstance(normalize_input, bool) and normalize_input) or isinstance(normalize_input, th.BoolTensor):
            self.normalize_input = True
            # lock = mp_context.Lock() if isinstance(mp_context, BaseContext) else None
            mask = normalize_input if isinstance(normalize_input, th.BoolTensor) else None
            self.norm = Standardize(num_features=self.input_size, center=False, mask=mask, synchronized=True)
            
    @property
    @abstractmethod
    def state(self) -> Optional[Any]:
        raise NotImplementedError()

    @state.setter
    @abstractmethod
    def state(self, value: Any):
        raise NotImplementedError()

    @abstractmethod
    def reset(self):
        raise NotImplementedError()

    @abstractmethod
    def action_probability_distribution(self, latent_state: th.Tensor) -> Distribution:
        raise NotImplementedError()

    @abstractmethod
    def step(self, state: th.Tensor, **kwargs):
        """

        Parameters
        ----------
        state : torch.Tensor
            Batch of observations: (N, observation_dim)
        Returns
        -------
        actions : torch.Tensor
            Batch of actions: (N, action_dim)
        log_prob : torch.Tensor
            Batch of log(probability of action): (N,)
        """
        raise NotImplementedError()

    @abstractmethod
    def act(self, state: th.Tensor, **kwargs):
        """
        
        Parameters
        ----------
        state : torch.Tensor
            Batch of observations: (N, observation_dim)
        Returns
        -------
        actions : torch.Tensor
            Batch of actions: (N, action_dim)
        """
        raise NotImplementedError()

    @abstractmethod
    def export(self) -> nn.Module:
        """
        Export for production a simple module which outputs only logits

        Returns
        -------
        module : nn.Module
            Simple module which takes state as input and outputs logits
        """
        raise NotImplementedError()

    def extra_repr(self) -> str:
        return (
            f'observation_space={self.observation_space}\n'
            f'action_space={self.action_space}\n'
            f'normalize_input={self.normalize_input}'
        )


class AgentActor(Agent):
    """
    Base class for actor
    """
    def __init__(self,
                 observation_space: spaces.Space,
                 action_space: spaces.Space,
                 actor_class: Type[BaseModel] = MLP,
                 actor_kwargs: Optional[Mapping] = dict(layers=(64, 64)),
                 normalize_input: Union[bool, th.BoolTensor] = True,
                 # mp_context: Optional[BaseContext] = None,
                 **kwargs):
        super().__init__(observation_space=observation_space, action_space=action_space,
                         normalize_input=normalize_input, **kwargs)
        actor_kwargs = {**actor_kwargs} if isinstance(actor_kwargs, Mapping) else {}
        actor_kwargs['input_shape'] = observation_space.shape
        self.actor_encoder = actor_class(**actor_kwargs)
        self.actor = nn.Linear(self.actor_encoder.output_size, self.output_size)
        BaseModel.init_weights_orthogonal(self.actor, gain=1.0)
        # Divide weights of the action layer by 100 according to the paper:
        #  [Marcin Andrychowicz et. al,
        #   What Matters In On-Policy Reinforcement Learning?
        #    A Large-Scale Empirical Study](https://arxiv.org/abs/2006.05990)
        # self.actor.weight.data.div_(100.0)
        
    @property
    def state(self) -> th.Tensor:
        actor_state = getattr(self.actor_encoder, 'state', None)
        if actor_state is None:
            actor_state = th.Tensor()
        return actor_state

    @state.setter
    def state(self, value: th.Tensor):
        if th.is_tensor(value) and (value.numel() > 0):
            setattr(self.actor_encoder, 'state', value)

    def reset(self):
        if callable(getattr(self.actor_encoder, 'reset', None)):
            self.actor_encoder.reset()

    def action_probability_distribution(self, actor_latent_state: th.Tensor) -> Distribution:
        logits = self.actor(actor_latent_state)
        return Categorical(logits=logits)
    
    def step(self, state: th.Tensor, **kwargs):
        if self.normalize_input:
            state = self.norm(state)
        actor_latent_state = self.actor_encoder(state)
        distribution = self.action_probability_distribution(actor_latent_state)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
        return action, log_prob

    def act(self, state: th.Tensor, **kwargs):
        if self.normalize_input:
            state = self.norm(state)
        actor_latent_state: th.Tensor = self.actor_encoder(state)
        logits: th.Tensor = self.actor(actor_latent_state)
        action = logits.argmax(dim=-1, keepdim=True).squeeze().numpy()
        return action
    
    def evaluate_actions(self, states: th.Tensor, actions: th.Tensor) -> th.Tensor:
        if self.normalize_input:
            states = self.norm(states)
        actor_latent_states = self.actor_encoder(states)
        distribution = self.action_probability_distribution(actor_latent_states)
        log_probs = distribution.log_prob(actions)
        return log_probs

    def export(self) -> nn.Module:
        m = nn.Sequential()
        m.add_module('actor_encoder', self.actor_encoder)
        m.add_module('actor', self.actor)
        return m


class AgentActorCritic(AgentActor):
    def __init__(self,
                 observation_space: spaces.Space,
                 action_space: spaces.Space,
                 actor_class: Type[BaseModel] = MLP,
                 actor_kwargs: Optional[Mapping] = dict(layers=(64, 64), activation_fn='tanh', compute_layer_activity=False),
                 critic_class: Optional[Type[BaseModel]] = None,
                 critic_kwargs: Optional[Mapping] = None,
                 normalize_input: Union[bool, th.BoolTensor] = True,
                 **kwargs):
        super().__init__(observation_space=observation_space, action_space=action_space,
                         actor_class=actor_class, actor_kwargs=actor_kwargs, normalize_input=normalize_input,
                         **kwargs)
        if critic_class is None:
            critic_class = actor_class
            critic_kwargs = actor_kwargs
        else:
            critic_kwargs = {**critic_kwargs} if isinstance(critic_kwargs, Mapping) else dict(layers=(64, 64))
        critic_kwargs['input_shape'] = observation_space.shape
        self.critic_encoder = critic_class(**critic_kwargs)
        self.critic = nn.Linear(self.critic_encoder.output_size, 1)
        BaseModel.init_weights_orthogonal(self.critic, gain=1.0)
        
    @property
    def state(self) -> th.Tensor:
        actor_state = getattr(self.actor_encoder, 'state', None)
        if actor_state is None:
            actor_state = th.Tensor()
        critic_state = getattr(self.critic_encoder, 'state', None)
        if critic_state is None:
            critic_state = th.Tensor()
        return th.cat((actor_state, critic_state), dim=-1)

    @state.setter
    def state(self, value: th.Tensor):
        if th.is_tensor(value) and (value.numel() > 0):
            actor_size = self.actor_encoder.state_size
            actor_state, critic_state = value[..., :actor_size], value[..., actor_size:]
            if actor_state.numel() > 0:
                setattr(self.actor_encoder, 'state', actor_state)
            if critic_state.numel() > 0:
                setattr(self.critic_encoder, 'state', critic_state)
                
    def reset(self):
        if callable(getattr(self.actor_encoder, 'reset', None)):
            self.actor_encoder.reset()
        if callable(getattr(self.critic_encoder, 'reset', None)):
            self.critic_encoder.reset()

    def step(self, state: th.Tensor, **kwargs) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """

        Parameters
        ----------
        state : torch.Tensor
            Batch of observations: (N, observation_dim)
        Returns
        -------
        actions : torch.Tensor
            Batch of actions: (N, action_dim)
        values : torch.Tensor
            Batch of values: (N,)
        log_prob : torch.Tensor
            Batch of log(probability of action): (N,)
        """
        if self.normalize_input:
            state = self.norm(state)
        actor_latent_state = self.actor_encoder(state)
        distribution = self.action_probability_distribution(actor_latent_state)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
        value_latent_state = self.critic_encoder(state)
        value = self.critic(value_latent_state).squeeze(-1)
        return action, value, log_prob

    def evaluate_actions(self, states: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Recompute action distribution (pi), values for the given states and log_probs of actions taken
        Parameters
        ----------
        states : torch.Tensor
            Batch of observations: (N, observation_dim)
        actions : torch.Tensor
            Batch of action taken by agent: (N,)
        Returns
        -------
        values : torch.Tensor
            Batch of values: (N,)
        log_prob : torch.Tensor
            Batch of log(probability of action): (N,)
        entropy : torch.Tensor
            Batch of entropies: (N,)
        """
        if self.normalize_input:
            states = self.norm(states)
        actor_latent_states = self.actor_encoder(states)
        distribution = self.action_probability_distribution(actor_latent_states)
        log_probs = distribution.log_prob(actions)
        value_latent_states = self.critic_encoder(states)
        values = self.critic(value_latent_states).squeeze(-1)
        return values, log_probs, distribution.entropy()

    def evaluate_states(self, states: th.Tensor) -> th.Tensor:
        """
        Compute only value function for a given batch of states
        Parameters
        ----------
        states : torch.Tensor
            Batch of observations: (N, observation_dim)
        Returns
        -------
        values : torch.Tensor
            Batch of values: (N,)
        """
        if self.normalize_input:
            states = self.norm(states)
        value_latent_states = self.critic_encoder(states)
        values = self.critic(value_latent_states).squeeze(-1)
        return values


def space_size(space: spaces.Space) -> int:
    if isinstance(space, spaces.Discrete):
        return int(space.n)
    elif isinstance(space, spaces.Box):
        return int(np.prod(space.shape))
    elif isinstance(space, spaces.MultiDiscrete):
        return int(np.sum(space.nvec))
    elif isinstance(space, spaces.MultiBinary):
        return int(space.n)
    elif isinstance(space, spaces.Dict):
        n = 0
        for k in space:
            n += space_size(space[k])
        return n
    else:
        raise ValueError(f'Invalid space: {repr(space)}')
