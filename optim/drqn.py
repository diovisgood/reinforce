from typing import Union, Optional
from collections import deque

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import timedelta
import numpy as np
import logging

from optim.autosaver import Autosaver


class ReplayBuffer(object):
    """
    Replay buffer.

    Parameters
    ----------
    size : int
        Max number of transitions to store in the buffer.
        When the buffer overflows the old memories are dropped.
    """
    
    def __init__(self, size):
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, state0, action, reward, state1, done):
        data = (state0, action, reward, state1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        states0, actions, rewards, states1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            state0, action, reward, state1, done = data
            states0.append(np.array(state0, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            states1.append(np.array(state1, copy=False))
            dones.append(done)
        return (
            np.array(states0),
            np.array(actions),
            np.array(rewards),
            np.array(states1),
            np.array(dones)
        )

    def sample(self, batch_size):
        """
        Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [
            random.randint(0, len(self._storage) - 1)
            for _ in range(batch_size)
        ]
        return self._encode_sample(idxes)


class ReplayMemory:
    def __init__(self, max_epi_num=50, max_epi_len=300):
        # capacity is the maximum number of episodes
        self.max_epi_num = max_epi_num
        self.max_epi_len = max_epi_len
        self.memory = deque(maxlen=self.max_epi_num)
        self.is_av = False
        self.current_epi = 0
        self.memory.append([])
        
    def reset(self):
        self.current_epi = 0
        self.memory.clear()
        self.memory.append([])
        
    def create_new_epi(self):
        self.memory.append([])
        self.current_epi = self.current_epi + 1
        if self.current_epi > self.max_epi_num - 1:
            self.current_epi = self.max_epi_num - 1

    def remember(self, state, action, reward):
        if len(self.memory[self.current_epi]) < self.max_epi_len:
            self.memory[self.current_epi].append([state, action, reward])

    def sample(self):
        epi_index = random.randint(0, len(self.memory)-2)
        if self.is_available():
            return self.memory[epi_index]
        else:
            return []

    def size(self):
        return len(self.memory)

    def is_available(self):
        self.is_av = True
        if len(self.memory) <= 1:
            self.is_av = False
        return self.is_av

    def print_info(self):
        for i in range(len(self.memory)):
            print('epi', i, 'length', len(self.memory[i]))


class DRQN2(nn.Module):
    def __init__(self, N_action):
        super(DRQN, self).__init__()
        self.lstm_i_dim = 16    # input dimension of LSTM
        self.lstm_h_dim = 16     # output dimension of LSTM
        self.lstm_N_layer = 1   # number of layers of LSTM
        self.N_action = N_action
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1)
        self.flat1 = Flatten()
        self.lstm = nn.LSTM(input_size=self.lstm_i_dim, hidden_size=self.lstm_h_dim, num_layers=self.lstm_N_layer)
        self.fc1 = nn.Linear(self.lstm_h_dim, 16)
        self.fc2 = nn.Linear(16, self.N_action)

    def forward(self, x, hidden):
        h1 = F.relu(self.conv1(x))
        h2 = self.flat1(h1)
        h2 = h2.unsqueeze(1)
        h3, new_hidden = self.lstm(h2, hidden)
        h4 = F.relu(self.fc1(h3))
        h5 = self.fc2(h4)
        return h5, new_hidden


class DRQN(Autosaver):
    """
    Asynchronous Advantage Actor-Critic (A3C) with Generalized Advantage Estimation (GAE)

    https://github.com/nicklashansen/a3c/blob/master/paper.pdf
    """
    
    def __init__(self,
                 env: gym.Env,
                 model: nn.Module,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 exploration_initial_eps: float = 0.1,
                 exploration_final_eps: float = 0.02,
                 gamma: float = 0.99,
                 gae_factor: float = 1.0,
                 value_factor: float = 0.5,
                 entropy_factor: float = 0.01,
                 repair_parameters=True,
                 scores_ema_period: int = 1000,
                 early_stopping=True,
                 tolerance=0.01,
                 patience=1000,
                 stat_interval: timedelta = timedelta(seconds=6),
                 step_delay: Optional[float] = None,
                 autosave_dir: Union[str, None] = '.',
                 autosave_prefix: Union[str, None] = None,
                 autosave_interval: Optional[timedelta] = timedelta(minutes=5),
                 log: Union[logging.Logger, str, None] = None):
        """

        Parameters
        ----------
        env : gym.Env
            Environment
        model : torch.nn.Module
            Model to be trained.
            Note: in this class the model parameters are always shared between worker processes.
        optimizer : torch.optim.Optimizer
            Torch optimizer.
        exploration_initial_eps : float
            Initial value of random action probability
        exploration_final_eps : float
            Final value of random action probability
        gamma : float
            Future discount factor for rewards (default: 0.99)
        gae_factor : float
            Generalized Advantage Estimation term coefficient (default: 1.0)
        value_factor : float
            Weight of value loss in total loss (default: 0.5)
        entropy_factor : float
            Weight of entropy loss in total loss (default: 0.01)
        repair_parameters : bool
            Sometimes during training some weights can become zero, NaN or infinitely large.
            Turn on this option to automatically detect and fix broken weights.
        scores_ema_period : int
            Period to compute `mean_score` value.
            Value `best_score` is a historical maximum of `mean_score` value.
        early_stopping : bool
            Stop training if error for test train_dataset didn't decrease at least for `tolerance` value
            for more than `patience` epochs.
        tolerance : float
            `best_score` value is updated only when `mean_score` is larger more than `tolerance`.
            Otherwise learning process will be stopped after `patience` iterations if `early_stopping` is True.
        patience : int
            For how many iterations to wait for model to improve `best_score`
            before stopping the training process. Default: 500
        stat_interval : Optional[timedelta]
            Interval to save scores to build chart. Default: timedelta(seconds=6)
        step_delay : Optional[float]
            Specify the required delay between steps of an episode in a worker process as the number of seconds.
            Default: None.
        """
        super().__init__(autosave_dir, autosave_prefix, autosave_interval, log)
        self.env = env
    
    def __init__(self, N_action, max_epi_num=50, max_epi_len=300):
        self.N_action = N_action
        self.max_epi_num = max_epi_num
        self.max_epi_len = max_epi_len
        self.drqn = DRQN(self.N_action)
        self.buffer = ReplayMemory(max_epi_num=self.max_epi_num, max_epi_len=self.max_epi_len)
        self.gamma = 0.9
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.drqn.parameters(), lr=1e-3)

    def remember(self, state, action, reward):
        self.buffer.remember(state, action, reward)

    def img_to_tensor(self, img):
        img_tensor = torch.FloatTensor(img)
        img_tensor = img_tensor.permute(2, 0, 1)
        return img_tensor

    def img_list_to_batch(self, x):
        # transform a list of image to a batch of tensor [batch size, input channel, width, height]
        temp_batch = self.img_to_tensor(x[0])
        temp_batch = temp_batch.unsqueeze(0)
        for i in range(1, len(x)):
            img = self.img_to_tensor(x[i])
            img = img.unsqueeze(0)
            temp_batch = torch.cat([temp_batch, img], dim=0)
        return temp_batch

    def train(self):
        if self.buffer.is_available():
            memo = self.buffer.sample()
            obs_list = []
            action_list = []
            reward_list = []
            for i in range(len(memo)):
                obs_list.append(memo[i][0])
                action_list.append(memo[i][1])
                reward_list.append(memo[i][2])
            obs_list = self.img_list_to_batch(obs_list)
            hidden = (Variable(torch.zeros(1, 1, 16).float()), Variable(torch.zeros(1, 1, 16).float()))
            Q, hidden = self.drqn.forward(obs_list, hidden)
            Q_est = Q.clone()
            for t in range(len(memo) - 1):
                max_next_q = torch.max(Q_est[t+1, 0, :]).clone().detach()
                Q_est[t, 0, action_list[t]] = reward_list[t] + self.gamma * max_next_q
            T = len(memo) - 1
            Q_est[T, 0, action_list[T]] = reward_list[T]

            loss = self.loss_fn(Q, Q_est)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def get_action(self, obs, hidden, epsilon):
        if random.random() > epsilon:
            q, new_hidden = self.drqn.forward(self.img_to_tensor(obs).unsqueeze(0), hidden)
            action = q[0].max(1)[1].data[0].item()
        else:
            q, new_hidden = self.drqn.forward(self.img_to_tensor(obs).unsqueeze(0), hidden)
            action = random.randint(0, self.N_action-1)
        return action, new_hidden


def get_decay(epi_iter):
    decay = math.pow(0.999, epi_iter)
    if decay < 0.05:
        decay = 0.05
    return decay


if __name__ == '__main__':
    random.seed()
    env = EnvTMaze(4, random.randint(0, 1))
    max_epi_iter = 30000
    max_MC_iter = 100
    agent = Agent(N_action=4, max_epi_num=5000, max_epi_len=max_MC_iter)
    train_curve = []
    for epi_iter in range(max_epi_iter):
        random.seed()
        env.reset(random.randint(0, 1))
        hidden = (Variable(torch.zeros(1, 1, 16).float()), Variable(torch.zeros(1, 1, 16).float()))
        for MC_iter in range(max_MC_iter):
            # env.render()
            obs = env.get_obs()
            action, hidden = agent.get_action(obs, hidden, get_decay(epi_iter))
            reward = env.step(action)
            agent.remember(obs, action, reward)
            if reward != 0 or MC_iter == max_MC_iter-1:
                agent.buffer.create_new_epi()
                break
        print('Episode', epi_iter, 'reward', reward, 'where', env.if_up)
        if epi_iter % 100 == 0:
            train_curve.append(reward)
        if agent.buffer.is_available():
            agent.train()
    np.save("len4_DRQN16_1e3_4.npy", np.array(train_curve))