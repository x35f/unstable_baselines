import os
from abc import abstractmethod

import torch
import gym.spaces
import numpy as np

from unstable_baselines.common import util


class BaseAgent(object):
    def __init__(self,**kwargs):
        super(BaseAgent,self).__init__(**kwargs)
        self.networks = {} # dict of networks, key = network name, value = network
    
    @abstractmethod
    def update(self,data_batch):
        pass

    @abstractmethod
    def select_action(self, state):
        pass

    def save_model(self, ite):
        save_dir = os.path.join(util.logger.log_path, 'models')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model_save_dir = os.path.join(save_dir, "ite_{}".format(ite))
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        for network_name, network in self.networks.items():
            save_path = os.path.join(model_save_dir, network_name + ".pt")
            torch.save(network, save_path)

    def load_model(self, model_dir):
        for network_name in self.networks.items():
            load_path = os.path.join(model_dir, network_name + ".pt")
            self.__dict__[network_name] = torch.load(load_path)
    
    def env_numpy_to_device_tensor(self, obs):
        """ Call it before need to pass data to networks.

        1. Transform numpy.ndarray to torch.Tensor;
        2. Make sure the tensor have the batch dimension;
        3. Pass the tensor to util.device;
        4. Make sure the type of tensor is float32.
        """
        device = util.device
        # util.debug_print(device)
        if not isinstance(obs, torch.Tensor):
            obs = torch.FloatTensor(obs)
            if len(obs.shape) < 2:
                obs = obs.unsqueeze(0)
        obs = obs.to(device)
        return obs

    def device_tensor_to_env_numpy(self, *args):
        """ Call it before need to pass data cpu.
        """
        return (item.detach().cpu().squeeze().numpy() for item in args)


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
