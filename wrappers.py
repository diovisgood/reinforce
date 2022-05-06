from typing import Sequence, OrderedDict, Dict, Optional, Mapping, Tuple, Any, Union, List
import numpy as np
import torch as th
import gym


class FlattenObservation(gym.ObservationWrapper):
    r"""Observation wrapper that flattens the observation."""

    def __init__(self, env):
        super(FlattenObservation, self).__init__(env)
        self.observation_space = gym.spaces.flatten_space(env.observation_space)

    def observation(self, observation):
        return gym.spaces.flatten(self.env.observation_space, observation)


class TorchWrapper(gym.ObservationWrapper):
    """Wrapper to convert gym state from numpy to torch tensor"""
    def __init__(self, env: gym.Env, device=None):
        super(TorchWrapper, self).__init__(env)
        self.device = device

    def observation(self, s):
        assert isinstance(s, np.ndarray)
        return th.tensor(s, dtype=th.get_default_dtype(), device=self.device).unsqueeze(dim=0)


class TorchFrameNorm(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, device=None):
        super().__init__(env)
        h, w, c = env.observation_space.shape
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(c, h, w), dtype=np.float32)
        self.device = device
    
    def observation(self, s: np.ndarray) -> th.Tensor:
        # Normalize image and convert it to torch format: (Batch, C, H, W)
        s = th.tensor(s, dtype=th.float32, device=self.device).permute(2, 0, 1) / 255.0
        s = s.unsqueeze(dim=0)
        return s


class TorchFrameWarp(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, width=84, height=84, grayscale=True):
        super().__init__(env)
        self.width = width
        self.height = height
        self.chan = 1 if grayscale else 3
        space = env.observation_space
        self.observation_space = gym.spaces.Box(
            low=space.low.min(), high=space.high.max(),
            shape=(self.chan, height, width),
            dtype=np.float32
        )
    
    def observation(self, s: th.Tensor) -> th.Tensor:
        b, c, h, w = s.shape
        s = s.to(dtype=th.float32)
        # Adjust image channels: RGB -> Grayscale or Grayscale -> RGB
        if (c > self.chan) and (self.chan == 1):
            s = s.mean(dim=1, keepdim=True)
        elif (c == 1) and (self.chan == 3):
            s = s.repeat(1, 3, 1, 1)
        # Resize image
        if (h != self.height) or (w != self.width):
            s = th.nn.functional.interpolate(s, (self.height, self.width), mode='bilinear')
        assert s.shape == (b, self.chan, self.height, self.width)
        return s


class TorchFrameStack(gym.Wrapper):
    def __init__(self, env: gym.Env, k: int):
        from collections import deque
        super().__init__(env)
        self.k = k
        self.frames = deque(maxlen=k)
        space = env.observation_space
        self.observation_space = gym.spaces.Box(
            low=space.low.min(), high=space.high.max(),
            shape=((space.shape[0] * k,) + space.shape[1:]),
            dtype=space.dtype
        )
    
    def reset(self, **kwargs):
        ob = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(ob)
        return th.cat(tuple(self.frames), dim=1)
    
    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return th.cat(tuple(self.frames), dim=1), reward, done, info


class NoopResetEnv(gym.Wrapper):
    """
    Sample initial states by taking random number of no-ops on reset.
    No-op is assumed to be action 0.
    Taken from https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
    """
    
    def __init__(self, env, noop_max=30):
        super().__init__(env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'
    
    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    """
    Return only every `skip`-th frame and repeat actions
    Taken from https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
    """
    def __init__(self, env: gym.Env, skip=4):
        super().__init__(env)
        # most recent raw observations (for max pooling across time steps)
        space = env.observation_space
        self.buffer = np.zeros((2,) + space.shape, dtype=space.dtype)
        self.skip = skip

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self.skip):
            obs, reward, done, info = self.env.step(action)
            if i == self.skip - 2:
                self.buffer[0] = obs
            if i == self.skip - 1:
                self.buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame doesn't matter
        max_frame = self.buffer.max(axis=0)

        return max_frame, total_reward, done, info
    

class FireResetEnv(gym.Wrapper):
    """
    Take action on reset for environments that are fixed until firing.
    Taken from https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
    """
    
    def __init__(self, env):
        super().__init__(env)
        assert len(env.unwrapped.get_action_meanings()) >= 3
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
    
    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs


class EpisodicLifeEnv(gym.Wrapper):
    """
    Make end-of-life == end-of-episode, but only reset on true game over.
    Done by DeepMind for the DQN and co. since it helps value estimation.
    Taken from https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
    """
    def __init__(self, env, loose_life_penalty: Optional[float] = -10):
        super().__init__(env)
        self.loose_life_penalty = loose_life_penalty
        self.lives = 0
        self.was_real_done = True
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if (lives < self.lives) and (lives > 0):
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
            if self.loose_life_penalty is not None:
                reward = self.loose_life_penalty
        self.lives = lives
        return obs, reward, done, info
    
    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class ResetOnEndOfLife(gym.Wrapper):
    """
    Make hard reset of a game if agent looses its life
    """
    
    def __init__(self, env, loose_life_penalty: Optional[float] = -10):
        super().__init__(env)
        self.loose_life_penalty = loose_life_penalty
        self.last_info = None

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        # lives = self.env.unwrapped.ale.lives()
        if (self.last_info is not None) and (self.last_info['ale.lives'] > info['ale.lives']):
            if self.loose_life_penalty is not None:
                reward = self.loose_life_penalty
            done = True
        self.last_info = info
        return state, reward, done, info
    
    def reset(self, **kwargs):
        super().reset(**kwargs)
        self.last_info = None


class ClipRewardEnv(gym.RewardWrapper):
    """
    Bin reward to {+1, 0, -1} by its sign..
    Taken from https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
    """
    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward):
        return np.sign(reward)
    