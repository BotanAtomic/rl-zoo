from typing import Any, SupportsFloat

import gymnasium as gym
import torch
from gymnasium.core import WrapperObsType, WrapperActType


class TorchWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super(TorchWrapper, self).__init__(env)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        state, info = self.env.reset(seed=seed, options=options)

        return torch.tensor(state, dtype=torch.float32), info

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        state, reward, terminated, truncated, info = self.env.step(action)
        return torch.tensor(state, dtype=torch.float32), reward, terminated, truncated, info
