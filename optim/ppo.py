"""
    Implementation of Proximal Policy Optimization (PPO)
    Copyright: Pavel B. Chernov, pavel.b.chernov@gmail.com
    Date: Dec 2020 - October 2021
    License: MIT
"""
from __future__ import annotations
from typing import Union, Sequence, Mapping, MutableMapping, Optional, List, Any, Literal, Callable, Tuple, Deque
from collections import deque
import logging
from datetime import timedelta, datetime
from numbers import Real
import copy
import time
import numpy as np
import torch as th
import torch.nn as nn
import torch.distributions
import torch.multiprocessing as mp
import torch.nn.functional as F
import gym

from optim.autosaver import Autosaver
from optim.scheduler import Scheduler
from optim.rollout_buffer import Rollout, RolloutsBuffer
from optim.agent import AgentActorCritic, space_size
from modules.standardize import Standardize


class PPO(Autosaver):
    """
    Proximal Policy Optimization (PPO)
    
    Features:
    - Generalized Advantage Estimator (GAE),
    - Detection and preventing of KL divergence
    - Support for agents with internal states (e.g.: Recurrent Neural Networks)
    
    Notes
    -----
    This implementation is based on the papers:

    - Original paper: https://arxiv.org/pdf/1707.06347.pdf
    - [Marcin Andrychowicz et. al, What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study](
    https://arxiv.org/abs/2006.05990)
    ## - [Parameter Space Noise for Exploration](https://arxiv.org/abs/1706.01905)
    
    Parameters
    ----------
    env : gym.Env
        Environment
    agent : AgentActorCritic
        Model to be trained.
        Model should return a tuple of three values: (logits, value, log_probs)
        `Logits` are used to choose action and `value` is used to estimate advantage.
        Note: in this implementation the model parameters are always shared between worker processes.
    optimizer : Optional[torch.optim.Optimizer]
        Torch optimizer.
        If None - `torch.optim.AdamW` is used.
    learning_rate : Union[float, Callable[[float], float]]
        The learning rate. It can be a function of the current progress (from 0 to 1).
        Default: Scheduler((3e-4, 1e-6), 's')
    weight_decay : Union[float, Callable[[float], float]]
        Weight decay for Adam optimizer.
        It can be a function of the current progress (from 0 to 1).
        Default: Scheduler((0.0, 1e-4), 's')
    n_workers : int
        The number of worker processes to use. Each worker will run its own environment. Default: 4
    max_episode_length : int
        The maximum number of steps to perform in an episode.
        If negative - run an episode without a limit.
        Default: -1
    steps_per_iteration : int
        How many steps each worker should perform between model weights update.
        If negative - it is the number of episodes a worker must collect.
        Default: 1024
    epochs_per_iteration : int
        Number of epochs when optimizing the surrogate loss. Default: 10
    reuse_rollouts : Optional[int]
        Specify the number of rollouts to be reused in the next iteration.
        Reusing trajectories enriches training samples
        and may help to avoid sharp dropdowns of gained performance.
        Set None to disable rollout reusing.
        Default: 2
    reuse_method : Literal['best', 'worse', 'extreme', 'random']
        The method to choose rollouts for reuse:
        - best - select rollouts with highest scores.
        - worse - select rollouts with lowest scores.
        - extreme - select both best and worse rollouts.
        - random - select some random rollouts.
        Default: 'best'
    reuse_decay : Union[float, Callable[[float], float]]
        Rollouts are chosen for reuse according to scores, agent gained in them.
        Depending on the `reuse_method` either best, worst, etc. - episodes can be selected for reuse.
        But we need to prevent a single episode to be reused forever.
        That is why we decay its score to a `mean_score` value, thus lowering its chances of being
        selected again.
        The speed of score decaying is determined by `reuse_decay` argument.
        At the beginning of training reusing can hurt learning, as algorithm can stuck in local minima.
        Reusing is most helpful at the end of training, when it prevents agent of forgetting the best
        ways to solve the task.
        That is why it is better to have fast decay at the beginning and low decay at the end of learning.
        Default: Scheduler((1.0, 0.0), 's')
    batch_size : int
        Minibatch size. Default: 64
    recurrent : bool
        If False - trains model on transitions: (state, action, reward, next_state),
        If True - trains model on trajectories: (transition1, transition2, ..., transitionN)
        Default: False
    future_discount_factor : float
        Future discount factor for rewards. Default: 0.99
    normalize_rewards : bool
        Whether to normalize or not rewards. Default: False
    normalize_returns : bool
        Whether to normalize or not returns. Default: False
    normalize_advantages : bool
        Whether to normalize or not advantages. Default: True
    advantage_smooth_factor : float
        Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
        Default: 1.0
    update_rollout_every_epoch : bool
        Recompute values of rollout states between each epoch of training.
        It is good to recompute internal states, values and advantages
        because agent weights have changed, and old advantages do not correspond to agent policy.
        This makes learning more stable though it takes significantly more time.
        Default: True
    value_factor : float
        Weight of value loss in total loss. Default: 0.5
        Value factor specifies the importance of estimating right value for a state.
    entropy_factor : float
        Weight of entropy loss in total loss. Default: 0.01
        Entropy factor prevents the model from always choosing the same actions.
        You can increase this factor to stimulate exploration.
    value_loss : Literal['huber', 'mse']
        Define which loss function to use:
        'mse' - Mean-Squared Error.
        'huber' - Huber loss.
        Default: 'mse'
    clip_range : Union[float, Callable[[float], float]]
        The clipping parameter for clipping policy loss into: [1.0 - `clip_range`, 1.0 + `clip_range`].
        It can be a function of the current progress (from 0 to 1).
        Default: 0.2
    limit_kl_divergence : Optional[float]
        Limit the KL divergence of action probabilities between updates,
        because the clipping with `clip_range` is not enough to prevent large drift.
        Default: 0.03
    max_grad_norm : float
        The maximum value for the gradient clipping. Default: 10.0
    repair_parameters : bool
        Sometimes during training some weights can become zero, NaN or infinitely large.
        Turn on this option to automatically detect and fix broken weights.
        In this case they are replaced with some small random values.
        Default: True
    scores_ema_period : int
        Period to compute `mean_score` attribute.
        Attribute `best_score` is a historical maximum of `mean_score` value.
        Default: 20
    measure_performance: Optional[Callable[[PPO], float]]
        You can specify a callback(PPO) to measure performance of a model.
        This performance will be saved in history and displayed on a chart.
        Default: None
    stat_interval : Optional[timedelta]
        Interval to save scores to build a chart. Default: timedelta(seconds=15)
    step_delay : Optional[float]
        Specify the required delay between steps of an episode in a worker process as the number of seconds.
        Use this parameter to reduce CPU usage and avoid overheating.
        Default: None.
    update_timeout : timedelta
        How long to wait for worker processes to complete their `steps_per_update` steps.
        Default: timedelta(minutes=5)
    autosave_dir : Optional[str]
        Specify directory to auto save checkpoints.
        When None is specified - no autosaving is performed.
        Default: '.'
    autosave_prefix : Optional[str]
        Specify file name prefix to be prepended to saved state, model, chart and conf files.
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
                 agent: AgentActorCritic,
                 optimizer: Optional[th.optim.Optimizer] = None,
                 learning_rate: Union[float, Callable[[float], float]] = Scheduler((3e-4, 1e-5), 's'),
                 weight_decay: Union[float, Callable[[float], float]] = Scheduler((0.0, 1e-4), 's'),
                 n_workers: int = 4,
                 max_episode_length: int = -1,
                 steps_per_iteration: int = -5,
                 epochs_per_iteration: int = 10,
                 reuse_rollouts: Optional[int] = None,
                 reuse_decay: Union[float, Callable[[float], float]] = Scheduler((1.0, 0.0), 's'),
                 reuse_method: Literal['best', 'worse', 'extreme', 'random'] = 'best',
                 batch_size: int = 64,
                 recurrent=False,
                 recurrent_sequence_length: int = 8,
                 future_discount_factor: float = 0.99,
                 normalize_rewards=False,
                 normalize_returns=False,
                 normalize_advantages=True,
                 advantage_smooth_factor: float = 0.95,
                 update_rollout_every_epoch=True,
                 value_factor: float = 0.5,
                 entropy_factor: float = 0.0,
                 value_loss: Literal['mse', 'huber'] = 'mse',
                 clip_range: Union[float, Callable[[float], float]] = 0.2,
                 limit_kl_divergence: Optional[float] = 0.04,
                 max_grad_norm: float = 0.5,
                 repair_parameters=True,
                 scores_ema_period: int = 20,
                 measure_performance: Optional[Callable[[PPO], float]] = None,
                 stat_interval: timedelta = timedelta(seconds=15),
                 step_delay: Optional[float] = None,
                 update_timeout: timedelta = timedelta(minutes=2),
                 autosave_dir: Optional[str] = '.',
                 autosave_prefix: Optional[str] = None,
                 autosave_interval: Optional[Union[int, timedelta]] = timedelta(minutes=5),
                 log: Union[logging.Logger, str, None] = None):
        super().__init__(autosave_dir, autosave_prefix, autosave_interval, log)
        self.env = env
        
        # Setup model and make it shared
        self.agent: AgentActorCritic = agent
        agent.share_memory()
        
        # Setup optimizer
        if optimizer is None:
            # optimizer = th.optim.RMSprop(agent.parameters(), lr=3e-5, momentum=0.9, eps=1e-4, centered=True)
            optimizer = th.optim.Adam(agent.parameters(), lr=3e-5, eps=1e-8, weight_decay=1e-4)
        self.optimizer: th.optim.Optimizer = optimizer
        
        # Setup parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.n_workers = n_workers
        self.max_episode_length = max_episode_length
        self.steps_per_iteration = steps_per_iteration
        self.epochs_per_iteration = epochs_per_iteration
        self.reuse_rollouts: Optional[int] = reuse_rollouts
        self.reuse_method = reuse_method
        self.reuse_decay = reuse_decay
        self.batch_size = batch_size
        self.recurrent = recurrent
        self.recurrent_sequence_length = recurrent_sequence_length
        self.future_discount_factor = future_discount_factor
        self.normalize_rewards = normalize_rewards
        self.normalize_returns = normalize_returns
        self.normalize_advantages = normalize_advantages
        self.advantage_smooth_factor = advantage_smooth_factor
        self.update_rollout_every_epoch = update_rollout_every_epoch
        self.value_factor = value_factor
        self.entropy_factor = entropy_factor
        self.clip_range = clip_range
        self.limit_kl_divergence: Optional[float] = limit_kl_divergence
        self.value_loss = value_loss
        self.max_grad_norm = max_grad_norm
        self.repair_parameters = repair_parameters
        self.scores_ema_period = max(1, scores_ema_period)
        self.scores_ema_factor = 2 / (scores_ema_period + 1)
        self.measure_performance = measure_performance
        self.stat_interval: timedelta = stat_interval
        self.step_delay: Optional[float] = step_delay
        self.update_timeout = update_timeout
        
        # Get multiprocess context
        self.mp = mp.get_context('spawn')

        # Rewards normalization module
        if normalize_rewards:
            self.norm_rewards: Optional[nn.Module] =\
                Standardize(1, momentum=2/500000, center=True, dim=-1, multiprocess_lock=self.mp.Lock())
        else:
            self.norm_rewards: Optional[nn.Module] = None

        # United rollout used to update model weights at one iteration
        self.buffer = RolloutsBuffer(normalize_returns=normalize_returns, normalize_advantages=normalize_advantages)

        # Queue to collect episode rollouts from worker processes
        self.rollout_queue = self.mp.Queue(maxsize=max(100, n_workers))

        # A register to count the number of workers which have finished collecting their rollout
        self.workers_done: mp.Value = self.mp.Value('i', 0)

        # Event used to signal worker processes that they can continue episode
        # and start collecting next rollout
        self.continue_event: mp.Event = self.mp.Event()

        # Event used to terminate worker processes
        self.terminate_event = self.mp.Event()
        
        # Total number of timesteps completed
        self.n_timesteps: mp.Value = self.mp.Value('i', 0)

        # Total number of episodes completed
        self.n_episodes: mp.Value = self.mp.Value('i', 0)

        # Number of model weights updates
        self.n_updates = 0
        
        # Keep history of all scores
        self.stat = {}
        self.best_score: Optional[Real] = None
        self.mean_score: Optional[Real] = None

        # This value will be set in train() to let worker processes know total timesteps
        self.total_timesteps = -1
        
        self.mean_policy_loss = np.inf
        self.mean_value_loss = np.inf
        self.mean_entropy_loss = np.inf

        # Try to load previously saved state
        self.autoload()
    
    @property
    def progress(self) -> float:
        return (self.n_timesteps.value / self.total_timesteps) if (self.total_timesteps > 0) else 1.0
    
    def train(self,
              total_timesteps: Optional[int] = None,
              seed: Optional[int] = None,
              render=False,
              **kwargs):
        """

        Parameters
        ----------
        total_timesteps : Optional[int]
            Specify maximum number of steps. Default: None.
        seed : Optional[Any]
            Specify any seed value to get consistently repeated episodes. Default: None.
        render : bool
            Display test run of model with `stat_interval`. Default: False.
        kwargs : Any
            Specify additional named arguments to pass to `env.reset(**kwargs)`
        """
        self.total_timesteps = total_timesteps
        
        # Set `train` mode for agent and normalization modules
        self.agent.train()
        if self.normalize_rewards:
            self.norm_rewards.train()

        # Not all environments do support `human` render mode
        render = (render and isinstance(getattr(self.env, 'metadata', None), Mapping)
                  and ('human' in self.env.metadata.get('render.modes', [])))
        
        # Get some initial seed
        if seed is None:
            seed = time.process_time_ns()

        # Instantiate worker processes
        processes = []
        for i in range(self.n_workers):
            p = self.mp.Process(
                target=self._collect_experience,
                kwargs=dict(seed=(seed + i), render=render, **kwargs)
            )
            p.start()
            processes.append(p)

        if self.log:
            self.log.info(f'Starting training with {self.n_workers} worker processes')
        
        try:
            # Monitor and control training process in a loop
            scores = deque(maxlen=self.scores_ema_period)
            reused_rollouts: List[Rollout] = []
            last_stat_time = datetime.now()
            while True:
                # Wait for workers processes to collect new experience
                # Each worker process should collect rollouts of `steps_per_iteration` in total
                # Also collect scores from each rollout and update `mean_score` and `best_score`
                self._wait_for_rollouts(scores)
                
                # Add some reused rollouts if any
                reuse_decay_factor = self.reuse_decay(self.progress) if callable(self.reuse_decay) else self.reuse_decay
                for rollout in reused_rollouts:
                    # Decay reused rollout score to the mean score.
                    # This prevents a single rollout to be kept in the buffer forever.
                    new_score = rollout.score * (1 - reuse_decay_factor) + self.mean_score * reuse_decay_factor
                    self.buffer.add(rollout._replace(score=new_score))

                # Finalize buffer. Normalize returns and advantages if needed
                self.buffer.finalize_buffer()

                # Perform `self.epochs_per_iteration` epochs of model training
                self._update_model()
                
                # Choose some episodes to be kept for the next iteration
                if isinstance(self.reuse_rollouts, int):
                    reused_rollouts = self.buffer.export(n=self.reuse_rollouts, method=self.reuse_method)
                
                # Empty buffer for new experience
                self.buffer.reset()
                
                # Signal worker processes that main process has finished updating model
                with self.workers_done.get_lock():
                    self.workers_done.value = 0

                if (datetime.now() - last_stat_time) > self.stat_interval:
                    last_stat_time = datetime.now()
                    # Measure performance if needed
                    performance = None
                    if callable(self.measure_performance):
                        performance = self.measure_performance(self)
                    # Write down some statistics
                    self._update_stat(scores, performance)
                    # Print debug info
                    if self.log:
                        lr, wd = self._get_learning_rate(), self._get_weight_decay()
                        self.log.info(
                            f'Timesteps: {self.n_timesteps.value} '
                            f'Episodes: {self.n_episodes.value} '
                            f'Updates: {self.n_updates} '
                            f'Mean score: {self.mean_score if (self.mean_score is not None) else -np.inf:g} '
                            f'Best score: {self.best_score if (self.best_score is not None) else -np.inf:g} '
                            f'Performance: {performance if (performance is not None) else -np.inf:g} '
                            f'policy_loss: {self.mean_policy_loss:g} '
                            f'value_loss: {self.mean_value_loss:g} '
                            f'entropy_loss: {self.mean_entropy_loss:g} '
                            f'learning_rate: {lr if (lr is not None) else np.nan:g} '
                            f'weight_decay: {wd if (wd is not None) else np.nan:g} '
                        )
                
                # Signal worker processes to continue for next iteration
                self.continue_event.set()
                self.continue_event.clear()
                
                # Update reused rollouts, including actions log_probs, as agent was updated
                for rollout in reused_rollouts:
                    self._update_rollout(rollout, update_log_probs=True)
                    
                # Perform autosave if needed
                self.autosave()
                
                # Check for timesteps
                if isinstance(total_timesteps, int) and (self.n_timesteps.value >= total_timesteps):
                    if self.log:
                        self.log.info(f'Reached total timesteps: {total_timesteps}. Stopping.')
                    break
                
                # Delay to reduce CPU usage
                time.sleep(0.1)

        finally:
            # Signal event to stop worker processes
            self.terminate_event.set()
            # Wait to worker processes to stop
            for p in processes:
                p.join(timeout=1)
                if p.is_alive():
                    p.terminate()

    def _collect_experience(self,
                            seed: Optional[int] = None,
                            render=False,
                            **kwargs):
        """
        This method runs inside a worker process and collects episode rollouts

        Parameters
        ----------
        seed : Optional[Any]
            Specify any seed value to get consistently repeated episodes. Default: None.
        render : bool
            If True calls `env.render(mode='human')`. Default: False.
        kwargs : Any
            Specify additional named arguments to pass to `env.reset(**kwargs)`
        """
        # Setup torch to reduce load on cpu by this worker process
        th.set_num_threads(1)
    
        # Get initial state
        if seed is not None:
            self.env.seed(seed)
        state: th.Tensor = self.env.reset(**kwargs)
    
        # Make a shallow copy of model, so to retain access to its shared weights
        agent = copy.copy(self.agent)
        agent.reset()

        # Variable to collect episodes scores
        score = 0.0
    
        # Reset arrays to collect rollout values
        n_iteration_episodes = 0
        n_episode_steps = 0
        n_iteration_steps = 0
        states = []
        internal_states = []
        actions = []
        log_probs = []
        values = []
        rewards = []
    
        while True:
            # Check for termination signal
            if self.terminate_event.is_set():
                self.env.close()
                return
        
            if render:
                self.env.render(mode='human')
        
            # Get internal state if any
            internal_state = agent.state
        
            # Get output from model
            with th.no_grad():
                action, value, log_prob = agent.step(state)
        
            # Execute action
            next_state, reward, done, info = self.env.step(action.item())
            if self.step_delay:
                time.sleep(self.step_delay)
            score += float(reward)

            # Normalize reward if needed
            if self.normalize_rewards:
                with th.no_grad():
                    reward = self.norm_rewards(th.tensor([reward], dtype=th.float32)).item()
                    
            # Save tensors for rollout
            states.append(state.squeeze(0))
            internal_states.append(internal_state.squeeze(0))
            actions.append(action)
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            state = next_state
        
            # Update counters
            n_episode_steps += 1
            n_iteration_steps += 1
            with self.n_timesteps.get_lock():
                self.n_timesteps.value += 1
        
            # If episode ends - pack and send rollouts to the main process
            if done or ((self.max_episode_length > 0) and (n_episode_steps >= self.max_episode_length)):
                # Compute final value or take zeroes
                if (not done) and (state is not None):
                    _, value, _ = agent.step(state)
                    final_value = value.detach()
                else:
                    final_value = th.zeros_like(values[0])
                
                if len(states) > 1:
                    # Prepare rollout
                    raw_returns, raw_advantages = self.compute_returns_and_advantage(
                        rewards, values, self.future_discount_factor, self.advantage_smooth_factor,
                        final_value=final_value
                    )
                
                    # Prepare internal states, as first internal state of an episode can be empty
                    istates = self.pack_internal_states(internal_states)
                    
                    # Send rollout to the main process
                    rollout = Rollout(
                        states=th.stack(states).share_memory_(),
                        internal_states=istates.share_memory_(),
                        actions=th.tensor(actions).share_memory_(),
                        log_probs=th.tensor(log_probs).share_memory_(),
                        rewards=th.tensor(rewards).share_memory_(),
                        raw_returns=raw_returns.share_memory_(),
                        raw_advantages=raw_advantages.share_memory_(),
                        score=score
                    )
                    self.rollout_queue.put(rollout)

                    # Clear rollout arrays
                    del rollout, raw_returns, raw_advantages

                    # Increment episodes counters
                    n_iteration_episodes += 1
                    with self.n_episodes.get_lock():
                        self.n_episodes.value += 1

                # Reset score and steps counters
                score = 0.0
                n_episode_steps = 0

                # Clear episode buffers
                states.clear()
                internal_states.clear()
                actions.clear()
                log_probs.clear()
                values.clear()
                rewards.clear()
                
                if ((self.steps_per_iteration <= 0) and (n_iteration_episodes >= abs(self.steps_per_iteration))) or \
                   ((self.steps_per_iteration > 0) and (n_iteration_steps >= self.steps_per_iteration)):
                    n_iteration_episodes = 0
                    n_iteration_steps = 0
                    # Send signal that this worker has done its part
                    with self.workers_done.get_lock():
                        self.workers_done.value += 1
                    # Wait for the signal that main process has finished updating model
                    self.continue_event.wait()
            
                # Get initial state
                state = self.env.reset(**kwargs)
            
                # Reset model
                if callable(getattr(agent, 'reset', None)):
                    agent.reset()

    def _wait_for_rollouts(self, scores: Deque):
        with th.no_grad():
            start_wait_time = datetime.now()
            while True:
                # Read all rollouts from queue
                while not self.rollout_queue.empty():
                    rollout: Rollout = self.rollout_queue.get(block=False)
                    self.buffer.add(rollout)
                    
                    # Update scores
                    score = rollout.score
                    self._update_mean_score(score)
                    if (self.best_score is None) or (self.best_score < self.mean_score):
                        self.best_score = self.mean_score
                    scores.append(score)

                    del rollout

                # Check if all workers have collected enough experience
                if self.workers_done.value >= self.n_workers:
                    break
                if (datetime.now() - start_wait_time) > self.update_timeout:
                    raise RuntimeError(f'Timeout waiting for worker processes to collect new rollouts')
                time.sleep(0.010)
            
    def _update_model(self):
        # Update learning_rate if needed
        progress = self.progress
        self._update_learning_rate(progress)
        
        # Get clip range
        if callable(self.clip_range):
            clip_range = self.clip_range(progress)
        elif isinstance(self.clip_range, float):
            clip_range = self.clip_range
        else:
            raise ValueError(f'Invalid clip_range: {self.clip_range}')
        
        policy_losses, value_losses, entropy_losses = [], [], []

        kl_divergence = False

        for epoch in range(self.epochs_per_iteration):
            # Recompute advantages and values once per epoch, if needed
            if self.update_rollout_every_epoch and (epoch > 1):
                self._update_values_and_advantages()
                
            approx_kl_divs = []
            sampler = self.buffer.sample(self.batch_size, self.recurrent, self.recurrent_sequence_length)
            for batch in sampler:
                # Get next batch of either transitions if `self.recurrent=False`
                # or trajectories (sequences) otherwise
                states, internal_states, actions, old_log_probs, returns, advantages = batch
                
                # Reset agent
                self.agent.reset()

                # Setup internal states if any
                if internal_states is not None:
                    self.agent.state = internal_states
                
                # Get new action distribution and log probabilities
                new_values, new_log_probs, entropy = self.agent.evaluate_actions(states, actions)
                
                # Compute policy (actor) loss
                ratio = th.exp(new_log_probs - old_log_probs)
                policy_loss1 = ratio * advantages
                policy_loss2 = ratio.clamp(1.0 - clip_range, 1.0 + clip_range) * advantages
                policy_loss = -th.min(policy_loss1, policy_loss2).mean()
                
                # Compute value estimation (critic) loss
                if self.value_loss == 'mse':
                    # value_loss = (returns - new_values).pow(2).mean()
                    value_loss = F.mse_loss(input=new_values, target=returns, reduction='mean')
                elif self.value_loss == 'huber':
                    value_loss = F.smooth_l1_loss(input=new_values, target=returns, reduction='mean')
                else:
                    raise ValueError()
                
                # Entropy loss
                entropy_loss = -entropy.mean()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())

                if self.limit_kl_divergence is not None:
                    # Calculate approximate form of reverse KL Divergence for early stopping
                    # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                    # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                    # and Schulman blog: http://joschu.net/blog/kl-approx.html
                    with th.no_grad():
                        log_ratio = new_log_probs - old_log_probs
                        approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                        approx_kl_divs.append(approx_kl_div)
                    
                    if approx_kl_div > 1.5 * self.limit_kl_divergence:
                        kl_divergence = True
                        if self.log:
                            self.log.debug(
                                f'Early stopping at epoch {epoch + 1}/{self.epochs_per_iteration} '
                                f'due to reaching max kl: {approx_kl_div:.2f}/{self.limit_kl_divergence:.2f}'
                            )
                        break

                # Zero gradients
                self.optimizer.zero_grad()
                
                # Compute gradients
                (policy_loss + value_loss * self.value_factor + entropy_loss * self.entropy_factor).backward()
                
                # Clip gradients by `max_grad_norm`
                torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=self.max_grad_norm)

                # Update optimizer state and model parameters
                self.optimizer.step()

                # Check and fix broken parameters if any
                if self.repair_parameters:
                    self.perform_repair_parameters(self.optimizer.param_groups)

                # Increment updates counter
                self.n_updates += 1
            
            if kl_divergence:
                break
        
        # Update mean loss values
        self.mean_policy_loss = np.mean(policy_losses)
        self.mean_value_loss = np.mean(value_losses)
        self.mean_entropy_loss = np.mean(entropy_losses)
    
    def _update_values_and_advantages(self):
        for i in range(self.buffer.rollouts_count):
            self._update_rollout(self.buffer.get_rollout(i))
        if self.normalize_advantages:
            self.buffer.renormalize_advantages()
            
    def _update_rollout(self, rollout: Rollout, update_log_probs=False):
        # Iterate through rollout (trajectory)
        internal_states, values, log_probs = [], [], []
        self.agent.reset()
        self.agent.state = rollout.internal_states[0].unsqueeze(0)
        internal_states.append(rollout.internal_states[0])
        for state, action in zip(rollout.states, rollout.actions):
            with th.no_grad():
                value, log_prob, entropy = self.agent.evaluate_actions(state.unsqueeze(0), action.unsqueeze(0))
            internal_states.append(self.agent.state.squeeze(0))
            values.append(value)
            log_probs.append(log_prob)
        # Update advantages
        raw_advantages = self._compute_advantages(
            rollout.rewards, values, self.future_discount_factor, self.advantage_smooth_factor,
            final_value=th.zeros_like(values[0])
        )
        # Prepare internal states, as first internal state of an episode can be empty
        istates = self.pack_internal_states(internal_states[:-1])
        # Update internal_states and advantages
        rollout.internal_states.copy_(istates)
        rollout.raw_advantages.copy_(raw_advantages)
        # Update log_probs if needed
        if update_log_probs:
            rollout.log_probs.copy_(th.tensor(log_probs))
    
    @staticmethod
    def compute_returns_and_advantage(rewards: List[th.Tensor],
                                      values: List[th.Tensor],
                                      future_discount_factor: float = 0.99,
                                      advantage_smooth_factor: float = 0.95,
                                      final_value: Optional[th.Tensor] = None,
                                      ) -> Tuple[th.Tensor, th.Tensor]:
        advantages = []
        advantage = 0

        if final_value is None:
            final_value = th.zeros_like(values[0])

        returns = []
        R = final_value

        next_value = final_value
        for i in reversed(range(len(rewards))):
            reward = rewards[i]
            value = values[i]
            # Returns
            R = reward + R * future_discount_factor
            returns.insert(0, R)
            # Advantages
            td_error = reward + future_discount_factor * next_value - value
            advantage = td_error + future_discount_factor * advantage_smooth_factor * advantage
            advantages.insert(0, advantage)
        # Compute returns
        advantages = th.tensor(advantages)
        # returns = advantages + th.tensor(values)
        returns = th.tensor(returns)
        return returns, advantages

    @staticmethod
    def _compute_returns(rewards: List[th.Tensor],
                         future_discount_factor: float,
                         final_value: Optional[th.Tensor] = None
                         ) -> th.Tensor:
        returns = []
        R = 0 if (final_value is None) else final_value
        for reward in reversed(rewards):
            R = reward + R * future_discount_factor
            returns.insert(0, R)
        returns = th.tensor(returns)
        return returns

    @staticmethod
    def _compute_advantages(rewards: List[th.Tensor],
                            values: List[th.Tensor],
                            future_discount_factor: float,
                            advantage_smooth_factor: float,
                            final_value: Optional[th.Tensor] = None
                            ) -> th.Tensor:
        advantages = []
        advantage = 0
    
        if final_value is None:
            final_value = th.zeros_like(values[0])
            
        next_value = final_value
        for i in reversed(range(len(rewards))):
            reward = rewards[i]
            value = values[i]
            td_error = reward + next_value * future_discount_factor - value
            advantage = td_error + advantage * future_discount_factor * advantage_smooth_factor
            advantages.insert(0, advantage)
            next_value = value
    
        advantages = th.tensor(advantages)
        return advantages

    @staticmethod
    def pack_internal_states(internal_states: List[th.Tensor]) -> th.Tensor:
        # Prepare internal states, as first internal state of an episode can be empty
        if (len(internal_states) > 1) and th.is_tensor(internal_states[0]) and th.is_tensor(internal_states[-1]):
            if internal_states[0].numel() < internal_states[-1].numel():
                internal_states[0] = th.zeros_like(internal_states[1])
            istates = th.stack(internal_states)
        else:
            istates = th.zeros((len(internal_states), 0))
        return istates

    def _update_learning_rate(self, progress: float):
        if callable(self.learning_rate):
            lr = self.learning_rate(progress)
        elif isinstance(self.learning_rate, float):
            lr = self.learning_rate
        else:
            raise ValueError(f'Invalid learning_rate: {self.learning_rate}')
        if callable(self.weight_decay):
            wd = self.weight_decay(progress)
        elif isinstance(self.weight_decay, float):
            wd = self.weight_decay
        else:
            raise ValueError(f'Invalid weight_decay: {self.weight_decay}')
        # if lr != self._get_learning_rate():
        for ids, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = lr
            param_group['weight_decay'] = wd

    def _get_learning_rate(self):
        learning_rate = None
        for param_group in self.optimizer.param_groups:
            learning_rate = param_group['lr']
            break
        return learning_rate

    def _get_weight_decay(self):
        weight_decay = None
        for param_group in self.optimizer.param_groups:
            weight_decay = param_group['weight_decay']
            break
        return weight_decay

    def _update_stat(self, scores: Sequence[float], performance: Optional[float] = None):
        """Update statistics using latest collected scores"""
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
            if 'perf' not in self.stat:
                self.stat['perf'] = []
            if performance is not None:
                self.stat['perf'].append(performance)
            if 'steps' not in self.stat:
                self.stat['steps'] = []
            self.stat['steps'].append(self.n_timesteps.value)

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
        with self.n_episodes.get_lock():
            n = self.n_episodes.value
            if (n == 0) or (self.mean_score is None):
                self.mean_score = score
            elif n < self.scores_ema_period:
                self.mean_score = (self.mean_score * n + score) / (n + 1)
            else:
                self.mean_score = self.mean_score * (1 - self.scores_ema_factor) + score * self.scores_ema_factor

    def _make_optimizer_shared(self, optimizer: th.optim.Optimizer):
        """Make optimizer internal state shared (for multiprocess learning)"""
        # Get any state and run it through model
        state = self.env.reset()
        logit, value = self.agent(state)
        # Make zero gradients and perform one step of optimizer
        optimizer.zero_grad()
        (value.mean() * 0.0 + logit.mean() * 0.0).backward()
        optimizer.step()
        optimizer.zero_grad()
        # Make optimizer state shared
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]
                if not isinstance(state, MutableMapping):
                    continue
                for name in state:
                    value = state[name]
                    if th.is_tensor(value):
                        # If value is tensor - make it shared
                        value.share_memory_()
                    elif isinstance(value, Real):
                        # If value is int or float - convert it into tensor with shared memory
                        value = th.tensor(value, device=p.device)
                        value.share_memory_()
                        state[name] = value
        # Reset model to clear internal state of recurrent layers, if any
        if callable(getattr(self.agent, 'reset', None)):
            self.agent.reset()

    def state_dict(self):
        return {
            'model': self.agent.state_dict(),
            'optimizer': self.optimizer.state_dict() if (self.optimizer is not None) else None,
            'norm_rewards': self.norm_rewards.state_dict() if (self.norm_rewards is not None) else None,
            'n_timesteps': self.n_timesteps.value,
            'n_episodes': self.n_episodes.value,
            'n_updates': self.n_updates,
            'stat': self.stat,
            'best_score': self.best_score,
            'mean_score': self.mean_score,
        }

    def load_state_dict(self, state: MutableMapping, strict=False):
        if strict:
            required_keys = self.state_dict().keys()
            assert all(k in required_keys for k in state),\
                AssertionError(str(required_keys)+'\n'+str(state.keys()))
        if 'model' in state:
            model_state = state['model']
            del state['model']
            self.agent.load_state_dict(model_state, strict=strict)
        if 'optimizer' in state:
            optimizer_state = state['optimizer']
            del state['optimizer']
            if (optimizer_state is not None) and (self.optimizer is not None):
                self.optimizer.load_state_dict(optimizer_state)
        if 'norm_rewards' in state:
            norm_rewards = state['norm_rewards']
            del state['norm_rewards']
            if (norm_rewards is not None) and (self.norm_rewards is not None):
                self.norm_rewards.load_state_dict(norm_rewards, strict=strict)
        if 'n_timesteps' in state:
            self.n_timesteps.value = state['n_timesteps']
            del state['n_timesteps']
        if 'n_episodes' in state:
            self.n_episodes.value = state['n_episodes']
            del state['n_episodes']
        # Use parent class method to load all other values
        super().load_state_dict(state=state, strict=False)

    def get_params(self, deep=True):
        return dict(
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            n_workers=self.n_workers,
            max_episode_length=self.max_episode_length,
            steps_per_iteration=self.steps_per_iteration,
            epochs_per_iteration=self.epochs_per_iteration,
            reuse_rollouts=self.reuse_rollouts,
            reuse_method=self.reuse_method,
            reuse_decay=self.reuse_decay,
            batch_size=self.batch_size,
            recurrent=self.recurrent,
            recurrent_sequence_length=self.recurrent_sequence_length,
            future_discount_factor=self.future_discount_factor,
            normalize_rewards=self.normalize_rewards,
            normalize_returns=self.normalize_returns,
            normalize_advantages=self.normalize_advantages,
            advantage_smooth_factor=self.advantage_smooth_factor,
            update_rollout_every_epoch=self.update_rollout_every_epoch,
            value_factor=self.value_factor,
            entropy_factor=self.entropy_factor,
            clip_range=self.clip_range,
            limit_kl_divergence=self.limit_kl_divergence,
            value_loss=self.value_loss,
            max_grad_norm=self.max_grad_norm,
            repair_parameters=self.repair_parameters,
            scores_ema_period=self.scores_ema_period,
            stat_interval=self.stat_interval,
            step_delay=self.step_delay,
            update_timeout=self.update_timeout,
            **super().get_params(deep)
        )

    @property
    def config(self) -> Optional[str]:
        return (
            '[Agent]\n' + repr(self.agent) + '\n' +
            '[Optimizer]\n' + repr(self.optimizer) + '\n' +
            '[Trainer]\n' + repr(self) + '\n'
        )

    def model_state(self):
        return self.agent.state_dict()
    
    def load_model_state(self, model_state: Any, strict=False):
        self.agent.load_state_dict(model_state, strict=strict)

    def draw_chart(self, ax1):
        if ('mean' not in self.stat) or (len(self.stat['mean']) < 2):
            return
        x = self.stat['steps']
        scores = self.stat[50]
        color_scores = 'tab:blue'
        # ax1.fill_between(x, self.stat[25], self.stat[75], color='red', alpha=0.2, linewidth=0, label='scores 25..75%')
        # ax1.plot(x, self.stat[50], label='median', color='red', linewidth=1, linestyle='--')
        ax1.scatter(x, scores, s=5.0, marker='X', label='median(scores)', color=color_scores)
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            smooth = lowess(scores, x, frac=1/4)
            ax1.plot(smooth[:, 0], smooth[:, 1], label='smooth(median(scores))', linewidth=1, color=color_scores)
        except ModuleNotFoundError:
            pass
        ax1.set_xlabel('steps')
        ax1.set_ylabel('score', color=color_scores)
        ax1.tick_params(axis='y', labelcolor=color_scores)
        ax1.grid(True, axis='both')
        ax1.legend(loc='upper left')

        if ('perf' in self.stat) and (len(self.stat['perf']) > 0):
            perf = self.stat['perf']
            color_perf = 'tab:red'
            ax2 = ax1.twinx()
            ax2.scatter(x, perf, s=5.0, marker='o', label='performance', color=color_perf)
            try:
                from statsmodels.nonparametric.smoothers_lowess import lowess
                smooth = lowess(perf, x, frac=1 / 4)
                ax2.plot(smooth[:, 0], smooth[:, 1], label='smooth(performance)', linewidth=1, color=color_perf)
            except ModuleNotFoundError:
                pass
            ax2.set_ylabel('performance', color=color_perf)
            ax2.tick_params(axis='y', labelcolor=color_perf)
            ax2.legend(loc='upper right')
