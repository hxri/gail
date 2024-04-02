from logging import ERROR
import gymnasium as gym
import torch
import numpy as np


gym.logger.set_level(ERROR)  # Ignore warnings from Gym logger


class CartPoleEnv:
    def __init__(self):
        self.env = gym.make('CartPole-v1')

    def reset(self, seed):
        # state = self.env.reset()
        state, info = self.env.reset(seed=seed)
        return torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0)  # Add batch dimension to state

    def step(self, action):
        state, reward, terminal, info, _ = self.env.step(action[0].detach().numpy())  # Remove batch dimension from action
        return torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0), reward, terminal  # Add batch dimension to state

    def seed(self, seed):
        return self.env.seed(seed)

    def close(self):
        self.env.close()

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space
