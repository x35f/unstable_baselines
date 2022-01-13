import os
from abc import abstractmethod
from unstable_baselines.common import util
import torch


class BaseAgent(object):
    def __init__(self,
            **kwargs):
        self.networks = {} # dict of networks, key = network name, value = network
    
    @abstractmethod
    def update(self,data_batch):
        pass

    @abstractmethod
    def select_action(self, state):
        pass
    
    def snapshot(self, timestamp):
        save_dir = os.path.join(util.logger.log_path, 'models')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model_save_dir = os.path.join(save_dir, "ite_{}".format(timestamp))
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        for network_name, network in self.networks.items():
            save_path = os.path.join(model_save_dir, network_name + ".pt")
            torch.save(network, save_path)

        def load_snapshot(self, load_dir):
            for network_name in self.networks.items():
                load_path = os.path.join(load_dir, network_name + ".pt")
                self.__dict__[network_name] = torch.load(load_path)


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
