from abc import abstractmethod
import torch
import os
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

    def save_model(self, target_dir, ite):
        target_dir = os.path.join(target_dir, "ite_{}".format(ite))
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        for network_name, network in self.networks.items():
            save_path = os.path.join(target_dir, network_name + ".pt")
            torch.save(network, save_path)

    def load_model(self, model_dir):
        for network_name, network in self.networks.items():
            load_path = os.path.join(model_dir, network_name + ".pt")
            network.load_state_dict(torch.load(load_path))


class RandomAgent(BaseAgent):
    def __init__(self,observation_space, action_space, **kwargs):
        self.observation_space = observation_space
        self.action_space = action_space
        from common.networks import VNetwork
        self.v_network = VNetwork(observation_space.shape[0], 1, [64, 64], reparameterize=False)

    def update(self,data_batch, **kwargs):
        return


    def select_action(self, state, **kwargs):
        return self.action_space.sample()


    def act(self, state, evaluate=False):
        return self.action_space.sample(), 1.

    def load_model(self, dir, **kwargs):
        pass
    

    def save_model(self, target_dir, ite, **kwargs):
        pass