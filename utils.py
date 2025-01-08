import os
import random

import gymnasium as gym
import numpy as np
import torch


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_state_dim(env: gym.Env) -> tuple[int, ...]:
    if isinstance(env.observation_space, gym.spaces.Discrete) or isinstance(env.observation_space, gym.spaces.Box):
        return env.observation_space.shape
    else:
        raise ValueError("Observation space not supported")


def get_action_dim(env: gym.Env) -> tuple[int, ...]:
    if isinstance(env.action_space, gym.spaces.Discrete):
        return (int(env.action_space.n),)
    elif isinstance(env.action_space, gym.spaces.Box):
        return env.action_space.shape
    else:
        raise ValueError("Action space not supported")
