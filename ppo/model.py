import torch
import torch.nn.functional as F
import os
from torch import nn
from common.models import BaseAgent
from common.networks import VNetwork, GaussianPolicyNetwork, get_optimizer
from common.buffer import ReplayBuffer
from common.rollout import rollout
import numpy as np
from common import util 

class SACAgent(torch.nn.Module, BaseAgent):
    def __init__(self,observation_space, action_space,
        gamma,
        **kwargs):
        state_dim = observation_space.shape[0]
        action_dim = action_space.shape[0]
        super(SACAgent, self).__init__()
        #save parameters
        self.args = kwargs
        #initilze networks
        self.v_network = VNetwork(state_dim, 1, **kwargs['v_network'])
        #self.target_value_network = VNetwork(state_dim + action_dim, 1,**kwargs['q_network'])
        self.policy_network = GaussianPolicyNetwork(state_dim,action_dim, action_space = action_space,  ** kwargs['policy_network'])

        #pass to util.device
        self.v_network = self.v_network.to(util.device)
        self.policy_network = self.policy_network.to(util.device)

        #initialize optimizer
        self.v_optimizer = get_optimizer(kwargs['v_network']['optimizer_class'], self.q1_network, kwargs['v_network']['learning_rate'])
        self.policy_optimizer = get_optimizer(kwargs['policy_network']['optimizer_class'], self.policy_network, kwargs['policy_network']['learning_rate'])

        #hyper-parameters
        self.gamma = gamma
        self.tot_update_count = 0 

    def update(self, data_batch):
        state_batch, action_batch, next_state_batch, reward_batch, done_batch = data_batch
        curr_state_v = self.v_network(state_batch)
        next_state_v = self.v_network(next_state_batch)


        delta = reward_batch + self.gamma * (1 - done_batch) * next_state_v - curr_state_v
        advantage = delta + self.gamma
        #compute value loss
        advantage = reward_batch + 
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        #compute policy loss=
        policy_loss = 0
        policy_loss_value = policy_loss.detach().cpu().numpy()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.tot_update_count += 1
        
        return {
            "loss/q1": q1_loss_value, 
            "loss/q2": q2_loss_value, 
            "loss/policy": policy_loss_value, 
            "loss/entropy": alpha_loss_value, 
            "others/entropy_alpha": alpha_value
        }
        

    def try_update_target_network(self):
        if self.tot_update_count % self.update_target_network_interval == 0:
            util.soft_update_network(self.q1_network, self.target_q1_network, self.target_smoothing_tau)
            util.soft_update_network(self.q2_network, self.target_q2_network, self.target_smoothing_tau)
            
    def select_action(self, state, evaluate=False):
        if type(state) != torch.tensor:
            state = torch.FloatTensor([state]).to(util.device)
        action, log_prob, mean = self.policy_network.sample(state)
        if evaluate:
            return mean.detach().cpu().numpy()[0]
        else:
            return action.detach().cpu().numpy()[0]


    def save_model(self, target_dir, ite):
        target_dir = os.path.join(target_dir, "ite_{}".format(ite))
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        #save q networks 
        save_path = os.path.join(target_dir, "Q_network_1.pt")
        torch.save(self.q1_network, save_path)
        save_path = os.path.join(target_dir, "Q_network_2.pt")
        torch.save(self.q2_network, save_path)
        #save policy network
        save_path = os.path.join(target_dir, "policy_network.pt")
        torch.save(self.policy_network, save_path)


    def load_model(self, model_dir):
        q1_network_path = os.path.join(model_dir, "Q_network_1.pt")
        self.q1_network.load_state_dict(torch.load(q1_network_path))
        q2_network_path = os.path.join(model_dir, "Q_network_2.pt")
        self.q2_network.load_state_dict(torch.load(q2_network_path))
        policy_network_path = os.path.join(model_dir, "policy_network.pt")
        self.policy_network.load_state_dict(torch.load(policy_network_path))


        



