# import unittest
from typing import Mapping, Optional
import os
import logging
import numpy as np
import torch
import gym
from datetime import timedelta

from optim.a3c import A3C
from optim.models import ModelConvGRU1, ModelFC1
from init import TorchWrapper

# Setup logging
logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(name)s %(message)s',
    datefmt='%y-%m-%d %H:%M:%S',
    level=logging.DEBUG,
)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
log = logging.getLogger('test')

CPU_LIMIT = max(1, os.cpu_count() // 2)


def space_size(space: gym.spaces.Space):
    if isinstance(space, gym.spaces.Box):
        return np.prod(space.shape)
    elif isinstance(space, gym.spaces.Discrete):
        return space.n
    elif isinstance(space, gym.spaces.MultiDiscrete):
        return np.sum(space.nvec)
    elif isinstance(space, gym.spaces.MultiBinary):
        return space.n
    else:
        raise ValueError()


class AtariRescale(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, size: int = 80):
        super(AtariRescale, self).__init__(env)
        self.size = size
        self.observation_space = gym.spaces.Box(0.0, 1.0, [1, size, size])

    def observation(self, frame):
        import cv2
        frame = frame[34:34 + 160, :160]
        frame = cv2.resize(frame, (self.size, self.size))
        frame = frame.mean(2, keepdims=True)
        frame = frame.astype(np.float32)
        frame = frame / 255.0
        frame = np.moveaxis(frame, -1, 0)
        return frame


class AtariAutoRestart(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_info: Optional[Mapping] = None

    def reset(self):
        self.last_info = None
        super().reset()
        # Press FIRE
        state, _, _, info = self.env.step(1)
        state, _, _, info = self.env.step(1)
        self.last_info = info
        return state

    def step(self, action):
        # if np.random.random() > 0.90:
        #     action = 1
        state, reward, done, info = self.env.step(action)
        # Check number of lives in ATARI environments
        if isinstance(info, Mapping) and ('ale.lives' in info):
            if (self.last_info is not None) and self.last_info['ale.lives'] > info['ale.lives']:
                done = True
            self.last_info = info
        return state, reward, done, info


class MountRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, self.reward(reward, observation), done, info

    def reward(self, r, s):
        return 10 if (r >= 0) else (s[:, 1] * 10) ** 2


#class TestA3C(unittest.TestCase):
class TestA3C:
    def test_CartPole(self):
        self._perform_test(env_name='CartPole-v0', name_prefix='a3c_cartpole8')

    def test_MountainCar(self):
        self._perform_test(env_name='MountainCar-v0', name_prefix='a3c_mountcar0')
        
    def test_MountainCarContinuous(self):
        self._perform_test(env_name='MountainCarContinuous-v0', name_prefix='a3c_mountcarcont0')

    def test_Breakout(self):
        self._perform_test(env_name='Breakout-v0', name_prefix='a3c_breakout0', max_iter=15000,
                           hidden_size=256, image_input=True, step_delay=0.08)
        
    @staticmethod
    def _perform_test(env_name: str,
                      name_prefix: str,
                      max_iter=None,
                      max_steps=1000,
                      hidden_size=32,
                      image_input=False,
                      scores_ema_period=50,
                      step_delay=None,
                      **kwargs):
        # Setup torch
        torch.set_num_threads(1)
        torch.set_default_dtype(torch.float32)
        torch.set_default_tensor_type(torch.FloatTensor)
        # Prepare environment and model
        env = gym.make(env_name)
        if image_input:
            env = AtariRescale(env)
        env = TorchWrapper(env)
        if env_name.startswith('Breakout-'):
            env = AtariAutoRestart(env)
        if env_name.startswith('MountainCar-'):
            env = MountRewardWrapper(env)
        log.info(f'Env: {env}')
        # Initialize model
        input_size = env.observation_space.shape
        output_size = space_size(env.action_space)
        if image_input:
            model = ModelConvGRU1(input_shape=input_size, layers=hidden_size, output_shape=(output_size, 1), normalize_input=False)
        else:
            model = ModelFC1(
                input_shape=input_size, layers=hidden_size, output_shape=(output_size, 1), normalize_input=False)
        log.info(f'Model: {model}')
        trainer = A3C(
            env=env,
            model=model,
            entropy_factor=1.0,
            early_stopping=False,
            scores_ema_period=scores_ema_period,
            step_delay=step_delay,
            autosave_dir='D:\\users\\diov\\Workspace\\strategy',
            autosave_prefix=name_prefix,
            autosave_interval=timedelta(minutes=1),
            log=log,
            **kwargs
        )
        log.info(f'Trainer: {trainer}')
        trainer.autoload()
        # score = trainer.test(render=True, max_steps=1000)
        # print(f'score={score}')
        trainer.fit(n_workers=2, total_timesteps=max_iter, max_steps=max_steps, render=True, **kwargs)
        trainer.autosave()


if __name__ == '__main__':
    #unittest.main()
    t = TestA3C()
    # t.test_CartPole()
    t.test_MountainCar()
