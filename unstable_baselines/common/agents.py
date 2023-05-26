import os
from abc import abstractmethod
from unstable_baselines.common import util
import torch


class BaseAgent(torch.nn.Module):
    def __init__(self,
            **kwargs):
        super(BaseAgent, self).__init__()

    @abstractmethod
    def update(self,data_batch):
        pass

    @abstractmethod
    def select_action(self, state):
        pass
    


class RandomAgent(BaseAgent):
    def __init__(self, observation_space, action_space, **kwargs):
        self.observation_space = observation_space
        self.action_space = action_space

    def update(self,data_batch, **kwargs):
        return


    def select_action(self, state, **kwargs):
        return self.action_space.sample()


    def select_action(self, state, deterministic=False):
        return self.action_space.sample(), 1.

    def load_model(self, dir, **kwargs):
        pass

    def save_model(self, target_dir, ite, **kwargs):
        pass
