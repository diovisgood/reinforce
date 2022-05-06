"""
    Implementation of Asynchronous Advantage Actor-Critic (A3C) with Generalized Advantage Estimation (GAE)
    Copyright: Pavel B. Chernov, pavel.b.chernov@gmail.com
    Date: Dec 2020 - July 2021
    License: MIT
"""
from typing import Union, Sequence, Mapping, MutableMapping, Optional, List, Any, Callable
from collections import deque
import logging
from datetime import timedelta, datetime
from numbers import Real
import copy
import torch as th
import torch.nn as nn
import torch.multiprocessing as mp
import torch.nn.functional as F
import numpy as np
import time
import gym

from optim.autosaver import Autosaver
from optim.shared import SharedAdamW
from optim.scheduler import Scheduler


class A3C(Autosaver):
    """
    Asynchronous Advantage Actor-Critic (A3C) with Generalized Advantage Estimation (GAE)
    
    Notes
    -----
    This implementation is based on the paper:
    https://github.com/nicklashansen/a3c/blob/master/paper.pdf
    
    Parameters
    ----------
    env : gym.Env
        Environment
    model : torch.nn.Module
        Model to be trained.
        Model should return a tuple of two values: (logits, value)
        `Logits` are used to choose action and `value` is used to estimate advantage.
        Note: in this implementation the model parameters are always shared between worker processes.
    optimizer : Optional[torch.optim.Optimizer]
        Torch optimizer.
        If None - `torch.optim.AdamW` is used.
    shared_optimizer : bool
        Specify True to make optimizer internal state shared between all worker processes.
        This may help to speed up model training.
        If False - each worker process will have its own instance of optimizer.
        Note: the model parameters are always shared in this implementation.
        Default: True
    learning_rate : Union[float, Callable[[float], float]]
        The learning rate. It can be a function of the current progress (from 0 to 1).
        Default: Scheduler((1e-3, 1e-6), 's')
    n_workers : int
        The number of worker processes to use. Each worker will run its own environment. Default: 4
    steps_per_update : int
        How many steps each worker performs between model weights update.
        If negative - update will be made only after episode completion.
        Default: 5
    future_discount_factor : float
        Future discount factor for rewards. Default: 0.99
    normalize_advantage : bool
        Whether to normalize or not the advantage. Default: False
    advantage_discount_factor : float
        Factor for trade-off of bias vs variance for Generalized Advantage Estimatorю
        Equivalent to classic advantage when set to 1.
        Default: 1.0
    value_factor : float
        Weight of value loss in total loss. Default: 0.5
        Value factor specifies the importance of estimating right value for a state.
    entropy_factor : float
        Weight of entropy loss in total loss. Default: 0.01
        Entropy factor prevents the model from always choosing the same actions.
        You can increase this factor to stimulate exploration.
    repair_parameters : bool
        Sometimes during training some weights can become zero, NaN or infinitely large.
        Turn on this option to automatically detect and fix broken weights.
        In this case they are replaced with some small random values.
        Default: True
    scores_ema_period : int
        Period to compute `mean_score` value.
        Value `best_score` is a historical maximum of `mean_score` value.
        Default: 20
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
                 shared_optimizer=True,
                 learning_rate: Union[float, Callable[[float], float]] = Scheduler((1e-3, 1e-6), 's'),
                 n_workers: int = 4,
                 steps_per_update: int = 5,
                 future_discount_factor: float = 0.99,
                 normalize_advantage=False,
                 advantage_discount_factor: float = 1.0,
                 value_factor: float = 0.5,
                 entropy_factor: float = 0.01,
                 max_grad_norm: float = 0.5,
                 repair_parameters=True,
                 scores_ema_period: int = 20,
                 stat_interval: timedelta = timedelta(seconds=15),
                 step_delay: Optional[float] = None,
                 autosave_dir: Optional[str] = '.',
                 autosave_prefix: Optional[str] = None,
                 autosave_interval: Optional[Union[int, timedelta]] = timedelta(minutes=5),
                 log: Union[logging.Logger, str, None] = None):
        super().__init__(autosave_dir, autosave_prefix, autosave_interval, log)
        self.env = env
        
        # Setup model and make it shared
        self.model = model
        model.share_memory()
        
        # Setup optimizer
        if optimizer is None:
            # optimizer = SharedRMSprop(model.parameters())
            # optimizer = SharedAdamW(model.parameters())
            # optimizer = th.optim.RMSprop(model.parameters())
            optimizer = SharedAdamW(model.parameters())
        else:
            # Make state of optimizer shared
            if shared_optimizer:
                self._make_optimizer_shared(optimizer)
        self.optimizer: th.optim.Optimizer = optimizer
        
        # Setup parameters
        self.learning_rate = learning_rate
        self.n_workers = n_workers
        self.steps_per_update = steps_per_update
        self.future_discount_factor = future_discount_factor
        self.normalize_advantage = normalize_advantage
        self.advantage_discount_factor = advantage_discount_factor
        self.value_factor = value_factor
        self.entropy_factor = entropy_factor
        self.max_grad_norm = max_grad_norm
        self.repair_parameters = repair_parameters
        self.scores_ema_period = max(1, scores_ema_period)
        self.scores_ema_factor = 2 / (scores_ema_period + 1)
        self.stat_interval: timedelta = stat_interval
        self.step_delay: Optional[float] = step_delay
        
        # Get multiprocess context
        self.mp = mp.get_context('spawn')
        
        # Mutex to control access to optimizer and model parameters
        self.lock = self.mp.Lock()
        
        # Queue to collect scores from worker processes
        self.scores_queue = self.mp.Queue(maxsize=scores_ema_period)
        
        # Event used to terminate worker processes
        self.terminate_event = self.mp.Event()
        
        # Total number of timesteps completed
        self.n_timesteps: mp.Value = self.mp.Value('i', 0)

        # Total number of episodes completed
        self.n_episodes: mp.Value = self.mp.Value('i', 0)

        # Number of iterations
        self.n_updates: mp.Value = self.mp.Value('i', 0)
        
        # Keep history of all scores
        self.stat = {}
        self.best_score: Optional[Real] = None
        self.mean_score: Optional[Real] = None

        # Count the # of iterations without improvement of `best_score`
        self.no_improvement_count = 0

        # This value will be set in fit() to let worker processes know total timesteps
        self.total_timesteps = -1

        # Try to load previously saved state
        self.autoload()
        
    def fit(self,
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
        self._update_learning_rate(self.n_timesteps.value / self.total_timesteps)

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
                target=self._train,
                kwargs=dict(seed=(seed + i), render=render, **kwargs)
            )
            p.start()
            processes.append(p)

        if self.log:
            self.log.info(f'Starting training with {self.n_workers} worker processes')
        
        try:
            # Monitor and control training process in a loop
            scores = deque(maxlen=self.scores_ema_period)
            last_stat_time = datetime.now()
            while True:
                # Read all scores from queue
                while not self.scores_queue.empty():
                    score = self.scores_queue.get(block=False)
                    self._update_mean_score(score)
                    if (self.best_score is None) or (self.best_score < self.mean_score):
                        self.best_score = self.mean_score
                    scores.append(score)

                if (datetime.now() - last_stat_time) > self.stat_interval:
                    last_stat_time = datetime.now()
                    # Write down some statistics
                    self._update_stat(scores)
                    # Print debug info
                    if self.log:
                        self.log.debug(
                            f'Timesteps: {self.n_timesteps.value} '
                            f'Episodes: {self.n_episodes.value} '
                            f'Updates: {self.n_updates.value} '
                            f'Mean score: {self.mean_score if (self.mean_score is not None) else -np.inf:g} '
                            f'Best score: {self.best_score if (self.best_score is not None) else -np.inf:g} '
                        )
                        
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

    def _train(self,
               seed: Optional[int] = None,
               render=False,
               **kwargs):
        """
        This method runs inside a worker process and trains shared model using shared optimizer
        
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

        # Get parameters
        gamma = self.future_discount_factor
        lambd = self.advantage_discount_factor
        value_factor = self.value_factor
        entropy_factor = self.entropy_factor

        # Get initial state
        if seed is not None:
            self.env.seed(seed)
        state = self.env.reset(**kwargs)

        # Make a shallow copy of model, so to retain access to its shared weights
        model = copy.copy(self.model)
        if callable(getattr(model, 'reset', None)):
            model.reset()
        model.train()

        # Reset arrays to collect episode values
        n_steps = 0
        score = 0.0
        values = []
        log_probs = []
        rewards = []
        entropies = []

        while True:
            # Check for termination signal
            if self.terminate_event.is_set():
                self.env.close()
                return
    
            if render:
                self.env.render(mode='human')
    
            # Get output from model
            qvalues, value = model(state)
    
            # Compute probabilities for actions: `prob` and also log(prob)
            prob = F.softmax(qvalues, dim=1)
            log_prob = F.log_softmax(qvalues, dim=1)
    
            # Compute entropy for current output of the model.
            # This will be a positive number, since `prob` contains numbers between 0 and 1,
            # and hence the log(prob) is negative.
            # Note that entropy is smaller when the probability distribution
            # is more concentrated on one action, so a larger entropy implies more exploration.
            # Thus we penalise small entropy, or equivalently, add -entropy to our loss.
            entropy = -(log_prob * prob).sum(dim=1)
    
            # Choose action
            # As said above, the entropy loss stimulates the model to use all possible actions.
            cat = th.distributions.Categorical(prob)
            action = cat.sample()
            log_prob = cat.log_prob(action)
            # action = prob.multinomial(num_samples=1).detach()
            # log_prob = log_prob.gather(dim=1, index=action)
            action = action.squeeze().numpy()
    
            # Execute action
            state, reward, done, info = self.env.step(action)
            if self.step_delay:
                time.sleep(self.step_delay)
    
            # Save tensors to later use them for update
            reward = th.tensor(reward, dtype=value.dtype, device=value.device)
            score = score + reward
            values.append(value.squeeze())
            log_probs.append(log_prob.squeeze())
            rewards.append(reward.squeeze())
            entropies.append(entropy.squeeze())
    
            # Increment shared timesteps counter
            with self.n_timesteps.get_lock():
                self.n_timesteps.value += 1
    
            # Sometimes update model weights, with `steps_per_update` interval
            n_steps += 1
            if done or ((self.steps_per_update > 0) and (n_steps >= self.steps_per_update)):
                n_steps = 0
                # Compute final value or take zeroes
                if (not done) and (state is not None):
                    _, value = model(state)
                    final_value = value.detach().squeeze()
                else:
                    final_value = th.zeros_like(values[0])
        
                # Compute loss
                returns = self._compute_returns(rewards, gamma)
                advantages = self._compute_advantages(rewards, values, gamma, trace_decay=lambd,
                                                      final_value=final_value, normalize=self.normalize_advantage)
                policy_loss = -(advantages.detach() * th.stack(log_probs)).mean()
                value_loss = F.smooth_l1_loss(input=th.stack(values), target=returns, reduction='mean')
                entropy_loss = th.stack(entropies).mean()
        
                # Update gradients in shared model
                with self.lock:
                    # Zero gradients in work model
                    self.optimizer.zero_grad()
                    # Compute gradients in work model
                    (policy_loss + value_loss * value_factor + entropy_loss * entropy_factor).backward()
                    # Clip gradients by `max_grad_norm`
                    th.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    # Update optimizer state and shared model parameters
                    self.optimizer.step()
                    # Check and fix broken parameters if any
                    if self.repair_parameters:
                        self.perform_repair_parameters(self.optimizer.param_groups)
                    # Update learning_rate if needed
                    self._update_learning_rate(self.n_timesteps.value / self.total_timesteps)

                # Increment updates counter
                with self.n_updates.get_lock():
                    self.n_updates.value += 1
        
                # Clear rollout arrays
                values.clear()
                log_probs.clear()
                rewards.clear()
                entropies.clear()
    
            if done:
                # Send score to the main process
                score = score.mean().item()
                self.scores_queue.put(score, block=False)
        
                # Increment shared episodes counter
                with self.n_episodes.get_lock():
                    self.n_episodes.value += 1
        
                # Get initial state
                state = self.env.reset(**kwargs)
        
                # Reset model
                if callable(getattr(model, 'reset', None)):
                    model.reset()
        
                # Reset score and arrays
                score = 0.0
                values.clear()
                log_probs.clear()
                rewards.clear()
                entropies.clear()

    @staticmethod
    def _compute_returns(rewards: List[th.Tensor],
                         future_discount_factor: float,
                         normalize=False
                         ) -> th.Tensor:
        returns = []
        R = 0
        for reward in reversed(rewards):
            R = reward + R * future_discount_factor
            returns.insert(0, R)
    
        returns = th.tensor(returns)
    
        if normalize:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    
        return returns

    @staticmethod
    def _compute_advantages(rewards: List[th.Tensor],
                            values: List[th.Tensor],
                            future_discount_factor: float,
                            trace_decay,
                            normalize=False,
                            final_value: Optional[th.Tensor] = None
                            ) -> th.Tensor:
        advantages = []
        advantage = 0
    
        if final_value is None:
            final_value = th.zeros_like(values[0])
    
        for i in reversed(range(len(rewards))):
            reward = rewards[i]
            value = values[i]
            td_error = reward + final_value * future_discount_factor - value
            advantage = td_error + advantage * future_discount_factor * trace_decay
            advantages.insert(0, advantage)
            final_value = value
    
        advantages = th.tensor(advantages)
    
        if normalize:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
        return advantages
    
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
        """
        # Make a work copy of shared model
        model = copy.deepcopy(self.model)
        model.eval()
    
        if seed is not None:
            self.env.seed(seed)
    
        # Get initial state
        while True:
            # Try to reset environment
            try:
                state = self.env.reset(**kwargs)
                break
            except (StopIteration, RuntimeError):
                pass
    
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

            # Get output from model
            qvalues, _ = model(state)

            # Select action
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
            if 'iter' not in self.stat:
                self.stat['iter'] = []
            self.stat['iter'].append(self.n_timesteps.value)

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
        logit, value = self.model(state)
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
        if callable(getattr(self.model, 'reset', None)):
            self.model.reset()

    def state_dict(self):
        return {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'n_timesteps': self.n_timesteps.value,
            'n_episodes': self.n_episodes.value,
            'n_updates': self.n_updates.value,
            'mean_score': self.mean_score,
            'best_score': self.best_score,
            'stat': self.stat,
        }
    
    def get_params(self, deep=True):
        return dict(
            learning_rate=self.learning_rate,
            n_workers=self.n_workers,
            steps_per_update=self.steps_per_update,
            future_discount_factor=self.future_discount_factor,
            normalize_advantage=self.normalize_advantage,
            advantage_discount_factor=self.advantage_discount_factor,
            value_factor=self.value_factor,
            entropy_factor=self.entropy_factor,
            max_grad_norm=self.max_grad_norm,
            repair_parameters=self.repair_parameters,
            scores_ema_period=self.scores_ema_period,
            stat_interval=self.stat_interval,
            step_delay=self.step_delay,
            **super().get_params(deep)
        )
    
    def load_state_dict(self, state: MutableMapping, strict=False):
        if strict:
            required_keys = self.state_dict().keys()
            assert all(k in required_keys for k in state),\
                AssertionError(str(required_keys)+'\n'+str(state.keys()))
        if 'model' in state:
            model_state = state['model']
            del state['model']
            self.model.load_state_dict(model_state, strict=strict)
        if 'optimizer' in state:
            optimizer_state = state['optimizer']
            del state['optimizer']
            if (optimizer_state is not None) and (self.optimizer is not None):
                self.optimizer.load_state_dict(optimizer_state)
        if 'n_timesteps' in state:
            self.n_timesteps.value = state['n_timesteps']
            del state['n_timesteps']
        if 'n_episodes' in state:
            self.n_episodes.value = state['n_episodes']
            del state['n_episodes']
        if 'n_updates' in state:
            self.n_updates.value = state['n_updates']
            del state['n_updates']
        # Use parent class method to load all other values
        super().load_state_dict(state=state, strict=False)

    def model_state(self):
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
