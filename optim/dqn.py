"""
    Implementation of Double Q-Learning with Prioritized Experience Replay
    Copyright: Pavel B. Chernov, pavel.b.chernov@gmail.com
    Date: Dec 2020 - July 2021
    License: MIT
"""
from typing import Union, Iterable, Sequence, Mapping, MutableMapping, Optional, Tuple, Any, Literal, Deque, List,\
    Callable
from collections import deque, namedtuple
import copy
import logging
from datetime import timedelta, datetime
from numbers import Real
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import gym

from optim.autosaver import Autosaver
from optim.buffer import ReplayBuffer, PrioritizedReplayBuffer
from optim.scheduler import Scheduler, s_curve


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class DQN(Autosaver):
    """
    [Double] [Dueling] Deep Q-Learning [with Prioritized Experience Replay]

    Notes
    -----
    This implementation is based on the papers:
    DQN: https://arxiv.org/abs/1312.5602
    Double-Q Learning: https://arxiv.org/abs/1509.06461
    Dueling DQN: https://arxiv.org/abs/1511.06581
    Prioritized Experience Replay: https://arxiv.org/abs/1511.05952
    
    Parameters
    ----------
    env : gym.Env
        Environment
    model : torch.nn.Module
        Model to be trained.
        In `dueling_mode` (the default), model should return a tuple(advantages, value).
        Where `advantages` - represent the *advantage* of choosing a specific action at a given state.
        And `value` - represents the *value* of a given state, regardless of the action taken.
        In this mode the action-state values, or Q-values, are computed as.
            Qvalues[action] = value + advantages[action] - mean(advantages)
        If `dueling_mode` is False, then algorithm works as vanilla DQN or double DQN.
        In this case model should only return Qvalues directly.
        Qvalues here represent the *value* of choosing a specific action at a given state.
        In any mode Qvalues are used to choose best actionas follows:
            action = argmax(Qvalues).
        Note: in this implementation the model parameters are always shared between worker processes.
    optimizer : Optional[torch.optim.Optimizer]
        Torch optimizer.
        If None - `torch.optim.Adam` is used.
    double_mode : bool
        If True algorithm will compute reference Q-value using two different models:
        both the online (aka `agent`) and the target (aka `target network`).
        Action for next state are chosen from online model as: a_next = argmax(Qonline(next_state)).
        While Q-value is chosen from target network: Q_next = Qtarget(next_state)[a_next].
        This way we decouple the action selection from the target Q value generation.
        Default: True
    dueling_mode: bool
        If True algorithm will compute reference Q-values using advantage.
        In `dueling_mode` (the default), model should return a tuple(advantages, value).
        Where `advantages` - represent the *advantage* of choosing a specific action at a given state.
        And `value` - represents the *value* of a given state, regardless of the action taken.
        In this mode the action-state values, or Q-values, are computed as.
            Qvalues[action] = value + advantages[action] - mean(advantages)
        Default: True
    learning_rate : Union[float, Callable[[float], float]]
        The learning rate. It can be a function of the current progress (from 0 to 1).
        Default: Scheduler((1e-3, 1e-6), 's')
    batch_size : int
        Number of samples from replay buffer to use for a single weights update.
        Default: 64
    gamma : float
        Future discount factor for rewards. Default: 0.99
    tau : float
        The soft update coefficient ("Polyak update", between 0 and 1).
        When 0 - no update of target network.
        When 1 - hard update, i.e. work models weights are fully copied into target network.
        Default: 1.0
    loss : Literal['huber', 'mse', 'reg']
        Define which loss function to use:
        'huber' - Huber loss.
        'mse' - Mean-Squared Error (as in Vanilla DQN paper).
        'reg' - DQNReg from the paper
                [Evolving Reinforcement Learning Algorithms](https://arxiv.org/abs/2101.03958)
        Default: 'mse'
    prioritized_replay : bool
        True - use limited buffer with sampling according to samples priorities.
        False - use simple limited FIFO buffer with uniform sampling.
        Default: True
    prioritized_replay_alpha : Optional[float]
        alpha parameter for prioritized replay buffer.
        It determines how much prioritization is used, with alpha=0 corresponding to the uniform case.
        Specify None to use uniform replay buffer.
        Default: 0.6
    replay_buffer_capacity : int
        The maximum number of samples (transitions) to store in the replay buffer.
        The bigger - the more useful samples can be kept, but the more memory is required.
        The smaller - the less memory is required, though learning is unstable due to small amount of samples.
        Default: 20000
    samples_to_start : int
        How many samples (transitions) should we collect into replay buffer before we start learning.
        Default: 10000
    samples_per_iteration : int
        How many new samples (transitions) should we get at least before making another model update.
        Default: 4
    updates_per_iteration : int
        How many updates of model weights to perform at each iteration.
        Default: 1
    target_update_period : int
        Number of iterations (updates of model parameters) after which target model is synchronized with work model.
        Default: 2500
    anneal_period : int
        The number of iterations (model updates) to decay probability from initial to final.
        This refers to `exploration` - as it goes from initial to final.
        And also this refers to the `beta` argument for prioritized sampling - as it goes from 0.0 to 1.0.
        Default: 10000
    exploration : Optional[Union[float, Callable[[float], float]]]
        Probability of choosing a random action (versus the best predicted one).
        It can be a function of the current progress (from 0 to 1).
        Use this argument to enable epsilon-greedy strategy for exploration/exploitation balance.
        If None - no epsilon-greedy strategy is used.
        Default: Scheduler((1.0, 0.05), 's')
    max_grad_norm : float
        The maximum value for the gradient clipping. Default: 10.0
    repair_parameters : bool
        Sometimes during training some weights can become zero, NaN or infinitely large.
        Turn on this option to automatically detect and fix broken weights.
        In this case they are replaced with some small random values.
        Default: True
    scores_ema_period : int
        Period to compute `mean_score` value.
        Value `best_score` is a historical maximum of `mean_score` value.
        Default: 10
    stat_interval : Optional[timedelta]
        Interval to save scores to build a chart. Default: timedelta(seconds=15)
    step_delay : Optional[float]
        Specify the required delay between steps of an episode in a worker process as the number of seconds.
        Use this parameter to reduce CPU usage and avoid overheating.
        Default: None.
    autosave_dir : Optional[str]
        Specify directory to auto save checkpoints.
        When None is specified - no autosaving is performed.
        Default: '.'
    autosave_prefix : Optional[str]
        Specify file name prefix for auto save checkpoints.
        When None is specified - no autosaving is performed.
        Default: None
    autosave_interval : Optional[Union[int, timedelta]]
        Specify interval for auto saving.
        When integer n value is specified: autosaving is performed after each n iterations.
        When timedelta t value is specified: autosaving is performed each t time interval.
        When None is specified: no autosaving is performed.
        Default: timedelta(minutes=5)
    log : Union[logging.Logger, str, None]
        Specify logging.Logger, str or None.
        You may specify a logging object or a name of logging stream to receive
        some debug, information or warnings from A3C instance.
        Default: None
    """
    
    def __init__(self,
                 env: gym.Env,
                 model: nn.Module,
                 optimizer: Optional[th.optim.Optimizer] = None,
                 double_mode: bool = True,
                 dueling_mode: bool = True,
                 learning_rate: Union[float, Callable[[float], float]] = Scheduler((1e-3, 1e-6), 's'),
                 batch_size: int = 64,
                 gamma: float = 0.99,
                 tau: float = 1.0,
                 loss: Literal['huber', 'mse', 'reg'] = 'mse',
                 anneal_period: int = 10000,
                 exploration: Optional[Union[float, Tuple[float, float]]] = Scheduler((1.0, 0.05), 's'),
                 prioritized_replay=True,
                 prioritized_replay_alpha: Optional[float] = 0.6,
                 replay_buffer_capacity: int = 100000,
                 samples_to_start: int = 1000,
                 samples_per_iteration: int = 4,
                 updates_per_iteration: int = 1,
                 target_update_period: int = 1000,
                 max_grad_norm: float = 10.0,
                 repair_parameters=True,
                 scores_ema_period: int = 10,
                 stat_interval: timedelta = timedelta(seconds=15),
                 step_delay: Optional[float] = None,
                 autosave_dir: Optional[str] = '.',
                 autosave_prefix: Optional[str] = None,
                 autosave_interval: Optional[Union[int, timedelta]] = timedelta(minutes=5),
                 log: Union[logging.Logger, str, None] = None):
        super().__init__(autosave_dir, autosave_prefix, autosave_interval, log)
        self.env = env

        # Setup model
        self.model = model

        # Setup optimizer
        if optimizer is None:
            # optimizer = th.optim.AdamW(model.parameters(), lr=learning_rate)
            # optimizer = th.optim.RMSprop(model.parameters())
            optimizer = th.optim.Adam(model.parameters())
        self.optimizer: th.optim.Optimizer = optimizer
        
        assert loss in {'huber', 'mse', 'reg'}
        
        # Setup parameters
        self.double_mode = double_mode
        self.dueling_mode = dueling_mode
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.loss = loss
        self.anneal_period = anneal_period
        self.exploration = exploration
        self.prioritized_replay = prioritized_replay
        self.prioritized_replay_alpha: Optional[float] = prioritized_replay_alpha
        self.replay_buffer_capacity = replay_buffer_capacity
        self.samples_to_start = samples_to_start
        self.samples_per_iteration = samples_per_iteration
        self.updates_per_iteration = updates_per_iteration
        self.target_update_period = target_update_period
        self.max_grad_norm = max_grad_norm
        self.repair_parameters = repair_parameters
        self.scores_ema_period = scores_ema_period
        self.scores_ema_factor = 2 / (scores_ema_period + 1)
        self.stat_interval: timedelta = stat_interval
        self.step_delay: Optional[float] = step_delay
        
        # Buffer to store sample transitions: (state, action, reward, next_state, done)
        if self.prioritized_replay:
            self.replay_buffer: Union[ReplayBuffer, PrioritizedReplayBuffer] =\
                PrioritizedReplayBuffer(replay_buffer_capacity, prioritized_replay_alpha)
        else:
            self.replay_buffer: Union[ReplayBuffer, PrioritizedReplayBuffer] =\
                ReplayBuffer(replay_buffer_capacity)

        # Total number of timesteps completed
        self.n_timesteps = 0

        # Total number of episodes completed
        self.n_episodes = 0
        
        # Number of model updates
        self.n_updates = 0
        
        # Last state of environment
        self._episode_state: Optional[th.Tensor] = None
        self._episode_score = 0.0

        # Keep history of all scores
        self.stat = {}
        self.best_score: Optional[Real] = None
        self.mean_score: Optional[Real] = None
        
        # Try to load previously saved state
        self.autoload()
        
    def fit(self,
            total_timesteps: int,
            seed: Optional[int] = None,
            render=False,
            **kwargs):
        """

        Parameters
        ----------
        total_timesteps : int
            Specify maximum number of timesteps.
        seed : Optional[Any]
            Specify any seed value to get consistently repeated episodes. Default: None.
        render : bool
            Display episodes on screen. Default: False.
        kwargs : Any
            Specify additional named arguments to pass to `env.reset(**kwargs)`
        """
        # Not all environments do support `human` render mode
        render = (render and isinstance(getattr(self.env, 'metadata', None), Mapping)
                  and ('human' in self.env.metadata.get('render.modes', [])))
        
        # Reset environment
        if seed is not None:
            self.env.seed(seed)
        self._episode_state = self.env.reset()
            
        # Make a target model to compute the reference Qvalues.
        target_model = copy.deepcopy(self.model)
        target_model.eval()

        # We will update weights of the agent, i.e.: self.model
        self.model.train()

        self._update_learning_rate((self.n_timesteps - self.samples_to_start) / total_timesteps)

        if self.log:
            self.log.info(f'Starting training for {total_timesteps=}')
    
        # Monitor and control training process in a loop
        scores = deque(maxlen=self.scores_ema_period)
        last_stat_time = datetime.now()
        last_target_update = self.n_timesteps
        while True:
            # Collect samples
            new_scores = []
            self._collect_samples(self.samples_per_iteration, self.replay_buffer, new_scores, render, **kwargs)
            if len(new_scores) > 0:
                scores.extend(new_scores)
                for score in new_scores:
                    self._update_mean_score(score)
                    if (self.best_score is None) or (self.mean_score > self.best_score):
                        self.best_score = self.mean_score

            # Sometimes save statistics and print debug messages
            if (datetime.now() - last_stat_time) > self.stat_interval:
                last_stat_time = datetime.now()
                self._update_stat(scores)
                # Print debug info
                if self.log:
                    self.log.debug(
                        f'Timesteps: {self.n_timesteps} '
                        f'Episodes: {self.n_episodes} '
                        f'Updates: {self.n_updates} '
                        f'Buffer: {len(self.replay_buffer)} '
                        f'Mean score: {self.mean_score if (self.mean_score is not None) else -np.inf:g} '
                        f'Best score: {self.best_score if (self.best_score is not None) else -np.inf:g} '
                        f'Exploration: {self._exploration():g}'
                    )
                    
            # Check if we have got enough samples to train on
            if len(self.replay_buffer) < max(self.samples_to_start, self.batch_size):
                continue

            # Perform training of a model
            progress = (self.n_timesteps - self.samples_to_start) / self.anneal_period
            self._update_model(target_model, progress)
            self._update_learning_rate(progress)

            # Periodically update target model parameters
            if self.n_timesteps - last_target_update >= self.target_update_period:
                if self.log:
                    self.log.debug(f'Updating target network (every {self.target_update_period} samples)')
                polyak_update(self.model.parameters(), target_model.parameters(), self.tau)
                last_target_update = self.n_timesteps

            # Perform autosave if needed
            self.autosave()
        
            # Check for max_samples
            if self.n_timesteps >= total_timesteps:
                if self.log:
                    self.log.info(f'Reached total timesteps: {total_timesteps}. Stopping.')
                break

        # Close any windows if any
        if render:
            self.env.close()

    def _collect_samples(self,
                         n: int,
                         buffer: Union[ReplayBuffer, PrioritizedReplayBuffer],
                         scores: Optional[Union[Deque, List]] = None,
                         render=False,
                         **kwargs) -> int:
        """
        Runs episodes and collects training samples
        Each sample is a transition: (state, action, reward, next_state, done)
        This method could be called in both synchronous or multiprocess environment.
        To use it in multiprocessing simply provide a mp.Queue for the `buffer` and `scores`.

        Parameters
        ----------
        n : int
            Specify required number of samples (=transitions) to collect.
            Default: None.
        buffer : Union[ReplayBuffer, PrioritizedReplayBuffer]
            A buffer to put samples into.
        scores : Optional[Union[Deque, List]]
            A list to put the resulting scores of episodes into.
            The scores of each episode is a sum of rewards received on each step of an episode.
            Specify None if you don't need to collect episode scores.
            Default: None
        render : bool
            If True calls `env.render(mode='human')`. Default: True.
        kwargs : Any
            Specify additional named arguments to pass to `env.reset(**kwargs)`
            
        Returns
        -------
        n_samples : int
            The number of samples collected
        """
        n_samples = 0
        while True:
            # Display window if needed
            if render:
                self.env.render(mode='human')
    
            # Select action
            qvalues = self._get_qvalues(self.model, self._episode_state)
            action = self._get_actions_best_or_random(qvalues.detach()).detach().numpy().item()
                
            # Execute action
            next_state, reward, done, info = self.env.step(action)
            if self.step_delay:
                time.sleep(self.step_delay)

            # Add sample
            sample = Transition(
                state=self._episode_state,
                action=action,
                reward=float(reward),
                next_state=next_state,
                done=done
            )
            buffer.put(sample)

            # Increment samples counter
            self.n_timesteps += 1
            n_samples += 1

            # Update score
            self._episode_score += reward

            # Get initial state
            if done:
                # Reset environment
                self._episode_state = self.env.reset(**kwargs)
                # Reset work model
                if callable(getattr(self.model, 'reset', None)):
                    self.model.reset()
                # Add episode score to the list
                if isinstance(scores, (List, Deque)):
                    scores.append(self._episode_score)
                # Reset score
                self._episode_score = 0.0
                # Increment episodes counter
                self.n_episodes += 1
            else:
                self._episode_state = next_state

            # Exit if we have collected enough samples
            if n_samples >= n:
                break
        
        return n_samples
    
    def _update_model(self, target_model: nn.Module, progress: float):
        """Performs one iteration of model weights update"""
        assert self.model.training
        for _ in range(self.updates_per_iteration):
            if self.prioritized_replay:
                beta = s_curve(progress)
                batch, weights, indexes = self.replay_buffer.sample(self.batch_size, beta)
            else:
                batch = self.replay_buffer.sample(self.batch_size)
                weights, indexes = None, None
            batch = Transition(*zip(*batch))
            states = th.cat(batch.state)
            actions = th.tensor(batch.action, device=states.device).unsqueeze(dim=-1).long()
            rewards = th.tensor(batch.reward, dtype=th.float32, device=states.device).unsqueeze(dim=-1)
            next_states = th.cat(batch.next_state)
            non_final_mask = 1.0 - th.tensor(batch.done, dtype=th.float32, device=states.device).unsqueeze(dim=-1)
            
            # Compute Q-values for specified actions in current states
            current_qvalues = self._get_qvalues(self.model, states)
            current_qvalues = th.gather(current_qvalues, dim=1, index=actions)
        
            # compute Q-values for next states with target network
            with th.no_grad():
                if self.double_mode:
                    # In Double DQN: actions are chosen with online model but Qvalues are computed with target model
                    next_qvalues = self._get_qvalues(self.model, next_states).detach()
                    next_actions = next_qvalues.max(dim=-1, keepdim=True)[1]
                    next_qvalues = self._get_qvalues(target_model, next_states).detach()
                    next_qvalues = next_qvalues.gather(dim=1, index=next_actions)
                else:
                    # In vanilla DQN: both actions and Qvalues are computed with target model
                    next_qvalues = self._get_qvalues(target_model, next_states).detach()
                    next_qvalues, _ = next_qvalues.max(dim=1, keepdim=True)
        
                # Compute target qvalues as specified by formula
                target_qvalues = rewards + self.gamma * next_qvalues * non_final_mask
            
            # Compute loss
            if self.loss == 'mse':
                loss = F.mse_loss(input=current_qvalues, target=target_qvalues, reduction='none')
            elif self.loss == 'huber':
                loss = F.smooth_l1_loss(input=current_qvalues, target=target_qvalues, beta=1.0, reduction='none')
            elif self.loss == 'reg':
                loss = dqn_reg_loss(input=current_qvalues, target=target_qvalues, reduction='none')
            else:
                raise ValueError(f'Invalid {self.loss=}')
    
            # Apply sample weights if prioritized replay is enabled
            if self.prioritized_replay:
                weights = th.tensor(weights, device=loss.device).unsqueeze(dim=-1)
                loss = (weights * loss).mean()
                # Update weights of samples
                errors = (target_qvalues - current_qvalues).detach().squeeze(dim=-1).numpy()
                self.replay_buffer.update(indexes, errors)
                del errors
            else:
                loss = loss.mean()
            assert th.isfinite(loss)
        
            # Perform one update iteration
            self.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            # Check and fix broken parameters if any
            if self.repair_parameters:
                self.perform_repair_parameters(self.optimizer.param_groups)
                
            # Increment parameters updates counter
            self.n_updates += 1
    
    def _get_qvalues(self, model: nn.Module, state: th.Tensor) -> th.Tensor:
        if self.dueling_mode:
            advantages, value = model(state)
            qvalues = value + advantages - advantages.mean(dim=1, keepdim=True)
        else:
            qvalues = self.model(state)
        return qvalues
    
    def _get_actions_best_or_random(self, qvalues: th.Tensor) -> th.Tensor:
        """
        Pick actions given their predicted `qvalues` using epsilon-greedy exploration strategy.
        This method is called from a worker processes which run episodes and collect samples.
        """
        # Get exploration probability (epsilon) for current iteration
        exploration = self._exploration()
        # Get best actions with the maximum qvalue
        best_actions = qvalues.max(dim=-1)[1]
        if (exploration is None) or (exploration <= 0):
            return best_actions
        # Prepare a batch of random actions
        batch_size, n_actions = qvalues.shape
        random_actions = th.randint(low=0, high=n_actions, size=(batch_size,), device=qvalues.device)
        # Choose best or random actions for the batch
        exploit_explore_prob = th.tensor([1 - exploration, exploration], dtype=th.float, device=qvalues.device)
        should_explore = th.multinomial(exploit_explore_prob, num_samples=batch_size, replacement=True).byte().to(
            device=qvalues.device)
        return th.where(should_explore, random_actions, best_actions)

    def _exploration(self) -> Optional[float]:
        """Returns exploration probability (epsilon) for current iteration"""
        if (self.exploration is None) or isinstance(self.exploration, float):
            return self.exploration
        elif callable(self.exploration):
            return self.exploration((self.n_timesteps - self.samples_to_start) / self.anneal_period)
        else:
            raise ValueError(f'Invalid exploration: {self.exploration}')

    def _update_learning_rate(self, progress: float):
        if callable(self.learning_rate):
            lr = self.learning_rate(progress)
        elif isinstance(self.learning_rate, float):
            lr = self.learning_rate
        else:
            raise ValueError(f'Invalid learning_rate: {self.learning_rate}')
        if lr != self._get_learning_rate():
            for ids, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = lr

    def _get_learning_rate(self):
        learning_rate = None
        for param_group in self.optimizer.param_groups:
            learning_rate = param_group['lr']
            break
        return learning_rate

    def test(self,
             seed: Optional[int] = None,
             max_steps: Optional[int] = None,
             render=True,
             **kwargs) -> Real:
        """
        This method runs one test episode and returns achieved score

        Parameters
        ----------
        seed : Optional[Any]
            Specify any seed value to get consistently repeated episodes. Default: None.
        max_steps : Optional[int]
            Specify maximum number of steps per episode. Default: None.
        render : bool
            If True calls `env.render(mode='human')`. Default: True.
        kwargs : Any
            Specify additional named arguments to pass to `env.reset(**kwargs)`
            
        Returns
        -------
        score : Real
            The sum of all rewards collected during the episode.
        """
        # Make a work copy of shared model
        model = copy.deepcopy(self.model)
        model.eval()
    
        if seed is not None:
            self.env.seed(seed)
    
        # Get initial state
        state = self.env.reset(**kwargs)

        # Reset work model
        if callable(getattr(model, 'reset', None)):
            model.reset()
    
        render = (render and isinstance(getattr(self.env, 'metadata', None), Mapping)
                  and ('human' in self.env.metadata.get('render.modes', [])))
    
        # Reset arrays to collect episode values
        n_steps = 0
        score = 0.0
    
        while True:
            # Display window if needed
            if render:
                self.env.render(mode='human')
        
            # Select action
            qvalues = self._get_qvalues(model, state)
            action = qvalues.argmax(dim=1).squeeze().numpy()
            
            # Execute action
            state, reward, done, info = self.env.step(action)
            if self.step_delay:
                time.sleep(self.step_delay)
        
            # Update score
            reward = th.tensor(reward, dtype=qvalues.dtype, device=qvalues.device)
            score = score + reward
        
            # Terminate episode if needed
            n_steps += 1
            if done or (isinstance(max_steps, int) and (n_steps >= max_steps)):
                break
    
        # Close any windows if any
        if render:
            self.env.close()
    
        # Return score
        score = score.mean().item()
        return score
    
    def _update_stat(self, scores: Sequence):
        """Write down some statistics based on latest scores"""
        percentiles = [25, 50, 75]
        if len(scores) >= len(percentiles):
            values = np.percentile(scores, q=percentiles)
            for percent, score in zip(percentiles, values):
                if percent not in self.stat:
                    self.stat[percent] = []
                self.stat[percent].append(score)
            if 'mean' not in self.stat:
                self.stat['mean'] = []
            self.stat['mean'].append(self.mean_score)
            if 'iter' not in self.stat:
                self.stat['iter'] = []
            self.stat['iter'].append(self.n_timesteps)

    def get_params(self, deep=True):
        return dict(
            double_mode=self.double_mode,
            dueling_mode=self.dueling_mode,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            gamma=self.gamma,
            anneal_period=self.anneal_period,
            exploration=self.exploration,
            prioritized_replay=self.prioritized_replay,
            prioritized_replay_alpha=self.prioritized_replay_alpha,
            replay_buffer_capacity=self.replay_buffer_capacity,
            samples_to_start=self.samples_to_start,
            samples_per_iteration=self.samples_per_iteration,
            updates_per_iteration=self.updates_per_iteration,
            target_update_period=self.target_update_period,
            max_grad_norm=self.max_grad_norm,
            repair_parameters=self.repair_parameters,
            scores_ema_period=self.scores_ema_period,
            stat_interval=self.stat_interval,
            step_delay=self.step_delay,
            **super().get_params(deep)
        )

    def perform_repair_parameters(self, param_groups: Sequence[Mapping[str, Any]]):
        """Check and replace zero, NaN or inf parameters with random values"""
        for group in param_groups:
            for param in group['params']:
                if isinstance(param, th.Tensor):
                    index = ((param.data != param.data) + (param.data == 0) +
                             (param.data == np.inf) + (param.data == -np.inf))
                    n = index.sum()
                    if n > 0:
                        if self.log:
                            self.log.warning(f'Repairing {n}/{param.numel()} bad parameters!')
                        param.data[index] = np.random.randn() / param.nelement()
                    index = ((param.data < -1e+10) + (param.data > 1e+10))
                    n = index.sum()
                    if n > 0:
                        if self.log:
                            self.log.warning(f'Clipping {n}/{param.numel()} huge parameters!')
                        param.data.clamp_(min=-1e+10, max=1e+10)
                        
    def _update_mean_score(self, score):
        """Update running mean of score"""
        n = self.n_episodes
        if (n == 0) or (self.mean_score is None):
            self.mean_score = score
        elif n < self.scores_ema_period:
            self.mean_score = (self.mean_score * n + score) / (n + 1)
        else:
            self.mean_score = self.mean_score * (1 - self.scores_ema_factor) + score * self.scores_ema_factor

    def state_dict(self):
        return {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'n_timesteps': self.n_timesteps,
            'n_episodes': self.n_episodes,
            'n_updates': self.n_updates,
            'mean_score': self.mean_score,
            'best_score': self.best_score,
            'stat': self.stat,
        }

    def load_state_dict(self, state: MutableMapping, strict=False):
        if strict:
            required_keys = self.state_dict().keys()
            assert all(k in required_keys for k in state), \
                AssertionError(str(required_keys) + '\n' + str(state.keys()))
        if 'model' in state:
            model_state = state['model']
            self.model.load_state_dict(model_state, strict=strict)
            del state['model']
        if 'optimizer' in state:
            optimizer_state = state['optimizer']
            if (optimizer_state is not None) and (self.optimizer is not None):
                self.optimizer.load_state_dict(optimizer_state)
            del state['optimizer']
        # Use parent class method to load all other values
        super().load_state_dict(state=state, strict=False)

    def model_state(self) -> Any:
        return self.model.state_dict()

    def load_model_state(self, model_state: Any, strict=False):
        self.model.load_state_dict(model_state, strict=strict)

    def draw_chart(self, ax1):
        if ('mean' not in self.stat) or (len(self.stat['mean']) < 2):
            return
        x = self.stat['iter']
        ax1.fill_between(x, self.stat[25], self.stat[75], color='red', alpha=0.2, linewidth=0, label='scores 25..75%')
        # ax1.plot(x, self.stat[50], label='median', color='red', linewidth=1, linestyle='--')
        ax1.plot(x, self.stat['mean'], label='ema(scores)', color='blue')
        ax1.set_xlabel('iter')
        ax1.set_ylabel('score', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.grid(True, axis='both')
        ax1.legend()


def polyak_update(
    params: Iterable[th.nn.Parameter],
    target_params: Iterable[th.nn.Parameter],
    tau: float,
) -> None:
    """
    Perform a Polyak average update on ``target_params`` using ``params``:
    target parameters are slowly updated towards the main parameters.
    ``tau``, the soft update coefficient controls the interpolation:
    ``tau=1`` corresponds to copying the parameters to the target ones whereas nothing happens when ``tau=0``.
    The Polyak update is done in place, with ``no_grad``, and therefore does not create intermediate tensors,
    or a computation graph, reducing memory cost and improving performance.  We scale the target params
    by ``1-tau`` (in-place), add the new weights, scaled by ``tau`` and store the result of the sum in the target
    params (in place).
    Parameters
    ----------
    params : Iterable[th.nn.Parameter]
        Parameters to use to update the target params
    target_params : Iterable[th.nn.Parameter]
        parameters to update
    tau : float
        the soft update coefficient ("Polyak update", between 0 and 1)
    """
    with th.no_grad():
        for param, target_param in zip(params, target_params):
            target_param.data.mul_(1 - tau)
            th.add(target_param.data, param.data, alpha=tau, out=target_param.data)


def dqn_reg_loss(input: th.Tensor,
                 target: th.Tensor,
                 weight=0.1,
                 reduction: Literal['mean', 'sum', 'none'] = 'mean'
                 ) -> th.Tensor:
    """
    In DQN, replaces Huber/MSE loss between train and target network
    Parameters
    ----------
    input : torch.Tensor
        Q(s[t], a[t]) of training network
    target : torch.Tensor
        Max Q value from the target network, including the reward and gamma:
        r + gamma * Q_target(s[t+1],a)
    weight : float
        weighted term that regularizes Q value.
        Paper defaults to 0.1 but theorizes that tuning this per env to some positive value may be beneficial.
    reduction : Literal['mean', 'sum', 'none']
        Default: 'mean'
    """
    # weight * Q(st, at) + delta^2
    delta = input - target
    loss = weight * input + th.pow(delta, 2)
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    return loss
