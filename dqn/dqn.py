import copy
from dataclasses import dataclass, field
from typing import Type

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from env.wrapper.torch_wrapper import TorchWrapper
from experience_logger.experience_logger import ExperienceLogger
from nn.simple_mlp import SimpleMLP
from replay_buffer import ReplayBuffer
from utils import get_state_dim, get_action_dim, seed_everything


@dataclass
class Config:
    # Environment settings
    env_name: str = "CartPole-v1"
    gamma: float = 0.99  # Discount factor
    seed: int = 42

    # Network settings
    q_network: Type[nn.Module] = None
    network_params: dict = field(default_factory=lambda: {"n_hidden_layers": 1, "hidden_dim": 128})
    lr: float = 1e-3  # Learning rate
    device: str = "cuda"  # "cpu" or "cuda"

    # Replay Buffer settings
    buffer_size: int = 10000  # Maximum size of the replay buffer
    minibatch_size: int = 128  # Mini-batch size for training
    replay_start_size: int = 1000  # Minimum buffer size before training begins

    # Epsilon Greedy settings
    initial_exploration: float = 0.9  # Initial epsilon value
    final_exploration: float = 0.05  # Final epsilon value
    final_exploration_frame: int = 10000  # Frames to decay epsilon to final value

    # Training settings
    training_steps: int = 60000  # Total number of training steps
    target_update: int = 1  # Frequency of target network updates (in steps)

    # Evaluation settings
    episode_eval_interval: int = 30000  # Interval (in steps) to evaluate the agent
    n_eval_episodes: int = 5  # Number of episodes for evaluation
    target_reward: float = 475  # Target reward to achieve
    consecutive_target_episodes: int = 5  # Number of consecutive episodes meeting the target

    # Logging settings
    log_dir: str = "logs"  # Directory for logs
    enable_logging: bool = True  # Toggle logging on or off


class DeepQNetwork:

    def __init__(self, config: Config):
        self.config: Config = config

        self.epsilon = config.initial_exploration

        test_env = gym.make(config.env_name)
        self.state_dim = get_state_dim(test_env)
        self.action_dim = get_action_dim(test_env)

        if len(self.action_dim) > 1:
            raise ValueError("Action Space is not discrete")

        self.replay_buffer = ReplayBuffer(
            buffer_size=config.buffer_size,
            batch_size=config.minibatch_size,
            state_dim=self.state_dim
        )

        test_env.close()

        print("Environment:", config.env_name)
        print(f"State Dimension: {self.state_dim}")
        print(f"Action Dimension: {self.action_dim}")

        self.q_network: nn.Module = config.q_network(
            self.state_dim, self.action_dim[0],
            **config.network_params
        ).to(config.device)

        self.q_target_network = copy.deepcopy(self.q_network).to(config.device)

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=config.lr, amsgrad=True)

        seed_everything(self.config.seed)

        self.env = TorchWrapper(gym.make(config.env_name))
        self.experience_logger = ExperienceLogger("dqn", enable=config.enable_logging)

        self.experience_logger.log_model(self.q_network, torch.zeros(1, *self.state_dim, device=config.device))

        self.current_episode = 0
        self.total_steps = 0

    def select_action(self, state: torch.Tensor, evaluation: bool = False) -> int:
        if self.epsilon > self.config.final_exploration:
            self.epsilon -= (self.config.initial_exploration -
                             self.config.final_exploration) / self.config.final_exploration_frame

        if not evaluation and np.random.rand() < self.epsilon:
            return np.random.choice(self.action_dim[0])
        else:
            state = state.unsqueeze(0).to(self.config.device)
            q_values = self.q_network(state)

            return q_values.argmax().item()

    def compute_loss(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor,
                     next_states: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:

        states = states.to(self.config.device)
        actions = actions.to(self.config.device)
        rewards = rewards.to(self.config.device)
        next_states = next_states.to(self.config.device)
        dones = dones.to(self.config.device)

        q_values = self.q_network(states)

        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            target_q_values = self.q_target_network(next_states)
            target_q_values = rewards + (self.config.gamma * target_q_values.max(1)[0] * (1 - dones))

        loss = torch.nn.functional.mse_loss(q_values, target_q_values)
        return loss

    def update_model(self):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample_buffer()
        loss = self.compute_loss(states, actions, rewards, next_states, dones)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.experience_logger.log_scalar("Train/Loss", loss.item(), self.current_episode)

        if self.total_steps % self.config.target_update == 0:
            self.q_target_network.load_state_dict(self.q_network.state_dict())

    def evaluate(self):
        evaluation_env = TorchWrapper(gym.make(self.config.env_name, render_mode="human"))
        rewards = []
        for _ in range(self.config.n_eval_episodes):
            state, _ = evaluation_env.reset()
            done = False
            episode_reward = 0.0

            while not done:
                action = self.select_action(state, True)
                state, reward, truncated, terminated, _ = evaluation_env.step(action)
                done = terminated or truncated
                episode_reward += reward

            rewards.append(episode_reward)

        evaluation_env.close()
        mean_reward = np.mean(rewards)
        self.experience_logger.log_scalar("Evaluation/Mean Reward", float(mean_reward), self.current_episode)

    def train(self) -> float:
        self.current_episode = 0
        self.total_steps = 1

        self.replay_buffer.reset()

        last_rewards = torch.zeros(self.config.consecutive_target_episodes, device=self.config.device)
        best_mean_reward = -np.inf

        while self.total_steps < self.config.training_steps:
            state, _ = self.env.reset()
            done = False
            episode_reward = 0.0

            while not done:
                action = self.select_action(state, False)
                next_state, reward, truncated, terminated, _ = self.env.step(action)
                done = terminated or truncated

                self.replay_buffer.store_transition(state, action, float(reward), next_state, done)
                episode_reward += reward
                state = next_state
                self.total_steps += 1

                if len(self.replay_buffer) >= self.config.minibatch_size and self.total_steps > self.config.replay_start_size:
                    self.update_model()

            last_rewards[self.current_episode % self.config.consecutive_target_episodes] = episode_reward
            mean_reward = last_rewards.mean().item()

            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward

            self.experience_logger.log_scalar("Train/Epsilon", self.epsilon, self.current_episode)
            self.experience_logger.log_scalar("Train/Reward", episode_reward, self.current_episode)
            self.experience_logger.log_scalar("Train/Steps", self.total_steps, self.current_episode)
            self.experience_logger.log_scalar("Train/Mean Reward", mean_reward, self.current_episode)

            print(f"Episode {self.current_episode}, mean reward: {mean_reward}", self.epsilon)

            if (self.current_episode >= self.config.consecutive_target_episodes
                    and 0 < self.config.target_reward < last_rewards.mean()):
                print(f"Target Reward Achieved in Episode {self.current_episode}, mean reward: {last_rewards.mean()}")
                self.evaluate()
                break

            self.current_episode += 1
            if self.current_episode % self.config.episode_eval_interval == 0 and self.current_episode > 0:
                self.evaluate()

        return last_rewards.mean().item()


if __name__ == "__main__":
    import optuna


    def objective(trial: optuna.Trial):
        config = Config(
            env_name="CartPole-v1",
            gamma=0.99,
            seed=42,

            q_network=SimpleMLP,
            network_params={"n_hidden_layers": 1, "hidden_dim": 128},
            lr=trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            device="cuda",

            buffer_size=trial.suggest_int("buffer_size", 5000, 500000),
            minibatch_size=trial.suggest_int("minibatch_size", 32, 256),
            replay_start_size=1000,

            initial_exploration=trial.suggest_float("initial_exploration", 0.5, 1.0),
            final_exploration=trial.suggest_float("final_exploration", 0.01, 0.1),
            final_exploration_frame=trial.suggest_int("final_exploration_frame", 10_000, 50_000),

            training_steps=60_000,
            target_update=trial.suggest_int("target_update", 1, 50),

            episode_eval_interval=30000,
            n_eval_episodes=5,
            target_reward=475,
            consecutive_target_episodes=5,

            log_dir="logs",
            enable_logging=True
        )
        dqn = DeepQNetwork(config)
        best_mean_reward = dqn.train()
        print("Best Mean Reward:", best_mean_reward)
        return best_mean_reward


    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)
    print(study.best_params)
    print(study.best_value)
