from typing import Sequence, OrderedDict, Dict, Optional, Mapping, Tuple, Any, Union, Type
import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import gym
import torch as th
import numpy as np
from datetime import timedelta, datetime, date
import logging
import warnings
from wrappers import FlattenObservation, TorchWrapper, TorchFrameNorm, TorchFrameWarp, TorchFrameStack,\
    MaxAndSkipEnv, NoopResetEnv, FireResetEnv, EpisodicLifeEnv, ResetOnEndOfLife, ClipRewardEnv

# Main directory to output intermediate files: saved checkpoints, charts and models
WORK_DIR = 'build'

# How many worker processes to use
N_WORKERS = 1  # max(1, os.cpu_count() // 2)

# Setup logging
logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(name)s %(message)s',
    datefmt='%y-%m-%d %H:%M:%S',
    level=logging.DEBUG,
)
# Turn off warnings from matplotlib
logging.getLogger('matplotlib').setLevel(logging.WARNING)
# Turn off UserWarning messages
warnings.filterwarnings('ignore')
# Initialize logger object
log = logging.getLogger('RL')

# Setup torch
DEVICE = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
th.set_num_threads(2)
th.set_default_dtype(th.float32)
th.set_default_tensor_type(th.FloatTensor)


def space_shape(space: gym.spaces.Space) -> Union[int, Tuple]:
    if isinstance(space, gym.spaces.Box):
        return tuple(space.shape)
    elif isinstance(space, gym.spaces.Discrete):
        return int(space.n)
    elif isinstance(space, gym.spaces.MultiDiscrete):
        return int(np.sum(space.nvec))
    elif isinstance(space, gym.spaces.MultiBinary):
        return int(space.n)
    else:
        raise ValueError()


class MountRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, self.reward(reward, observation), done, info

    def reward(self, r, s):
        return 10 if (r >= 0) else (s[:, 1] * 10) ** 2


def convert_action(action: th.Tensor, space: gym.spaces.Space) -> Any:
    if isinstance(space, gym.spaces.Box):
        return action.detach().squeeze(dim=0).tolist()
    elif isinstance(space, gym.spaces.Discrete):
        return action.detach().squeeze(dim=0).softmax(dim=0).argmax().item()
    else:
        raise ValueError()


def get_env(env_name: str, no_frame_stack=False) -> gym.Env:
    env = gym.make(env_name)
    
    if isinstance(env.observation_space, gym.spaces.Dict):
        env = FlattenObservation(env)
    
    is_atari = hasattr(gym.envs, 'atari') and isinstance(env.unwrapped, gym.envs.atari.atari_env.AtariEnv)

    if is_atari:
        env = NoopResetEnv(env, noop_max=30)
        # env = MaxAndSkipEnv(env, skip=4)
        if len(env.observation_space.shape) == 3:
            env = EpisodicLifeEnv(env)
            # env = ResetOnEndOfLife(env)
            if 'FIRE' in env.unwrapped.get_action_meanings():
                env = FireResetEnv(env)
            # env = ClipRewardEnv(env)
            env = TorchFrameNorm(env)
            env = TorchFrameWarp(env, width=64, height=64, grayscale=True)
            if not no_frame_stack:
                env = TorchFrameStack(env, k=2)
    else:
        env = TorchWrapper(env)
        
    # short_name = env_name.split('-', 1)[0].lower()
    
    return env


def ensure_dir_exists(dir: str):
    if not os.path.exists(dir):
        os.mkdir(dir)
    assert os.path.isdir(dir)


def get_new_output_dir(alg_name: str, env_name: str) -> str:
    alg_name, env_name = alg_name.lower(), env_name.rsplit("-", 1)[0].lower()
    prefix = f'{alg_name}_{env_name}_{datetime.now().strftime("%y%m%d%H%M")}'
    output_dir = os.path.join(WORK_DIR, prefix)
    assert not os.path.isdir(output_dir)
    os.mkdir(output_dir)
    return output_dir


def get_last_output_dir(alg_name: str, env_name: str) -> Optional[str]:
    alg_name, env_name = alg_name.lower(), env_name.rsplit("-", 1)[0].lower()
    latest_output_dir, latest_datetime = None, None
    for dir_name in os.listdir(WORK_DIR):
        output_dir = os.path.join(WORK_DIR, dir_name)
        if not os.path.isdir(output_dir):
            continue
        parts = dir_name.split('_')
        if len(parts) != 3:
            continue
        if (parts[0] != alg_name) or (parts[1] != env_name) or (not parts[2].isdigit()):
            continue
        if (latest_datetime is None) or (latest_datetime < int(parts[2])):
            latest_output_dir = output_dir
            latest_datetime = int(parts[2])
    return latest_output_dir
