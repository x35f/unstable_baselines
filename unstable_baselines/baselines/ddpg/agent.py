import torch
import torch.nn.functional as F
import gym 
import os
from torch import nn
from unstable_baselines.common.agents import BaseAgent
from unstable_baselines.common.networks import MLPNetwork, PolicyNetworkFactory, get_optimizer
from unstable_baselines.common.buffer import ReplayBuffer
import numpy as np
from unstable_baselines.common import util, functional

class DDPGAgent(torch.nn.Module, BaseAgent):
    def __init__(self,observation_space, action_space,
        target_smoothing_tau,
        **kwargs):
        obs_dim = observation_space.shape[0]
        action_dim = action_space.shape[0]
        super(DDPGAgent, self).__init__()

        #initilze networks
        self.q_network = MLPNetwork(obs_dim + action_dim, 1, **kwargs['q_network'])
        self.target_q_network = MLPNetwork(obs_dim + action_dim, 1,**kwargs['q_network'])
        self.policy_network = PolicyNetworkFactory.get(obs_dim, action_space,  ** kwargs['policy_network'])
        self.target_policy_network = PolicyNetworkFactory.get(obs_dim, action_space,  ** kwargs['policy_network'])

        #sync network parameters
        functional.soft_update_network(self.q_network, self.target_q_network, 1.0)
        functional.soft_update_network(self.policy_network, self.target_policy_network, 1.0)

        #pass to util.util.device
        self.q_network = self.q_network.to(util.device)
        self.target_q_network = self.target_q_network.to(util.device)
        self.policy_network = self.policy_network.to(util.device)
        self.target_policy_network = self.target_policy_network.to(util.device)
        
        #register networks
        self.networks = {
            'q_network': self.q_network,
            'target_q_network': self.target_q_network,
            'policy_network': self.policy_network,
            'target_policy_network': self.target_policy_network
        }

        #initialize optimizer
        self.q_optimizer = get_optimizer(kwargs['q_network']['optimizer_class'], self.q_network, kwargs['q_network']['learning_rate'])
        self.policy_optimizer = get_optimizer(kwargs['policy_network']['optimizer_class'], self.policy_network, kwargs['policy_network']['learning_rate'])

        #hyper-parameters
        self.gamma = kwargs['gamma']
        self.target_smoothing_tau = target_smoothing_tau

    def update(self, data_batch):
        obs_batch = data_batch['obs']
        action_batch = data_batch['action'] 
        next_obs_batch = data_batch['next_obs'] 
        reward_batch = data_batch['reward']
        done_batch = data_batch['done']
        
        curr_state_q_value = self.q_network(torch.cat([obs_batch, action_batch], dim=1))
        
        #get new action output
        curr_state_action_info = self.policy_network.sample(obs_batch)
        new_curr_state_action = curr_state_action_info['action_scaled']
        next_state_action_info = self.target_policy_network.sample(next_obs_batch)
        next_state_action = next_state_action_info['action_scaled']


        next_state_q_value = self.target_q_network(torch.cat([next_obs_batch, next_state_action], dim=1))
        target_q = reward_batch + self.gamma * (1. - done_batch) * next_state_q_value


        #compute q loss
        q_loss = F.mse_loss(curr_state_q_value, target_q.detach())

        q_loss_value = q_loss.item()
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        #compute policy loss
        new_curr_state_q_value = self.q_network(torch.cat([obs_batch, new_curr_state_action], dim=1))
        policy_loss = - new_curr_state_q_value.mean()
        policy_loss_value = policy_loss.item()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()

        self.policy_optimizer.step()

        #update target network
        self.update_target_network()

        return {
            "loss/q": q_loss_value, 
            "loss/policy": policy_loss_value, 
        }
        

    def update_target_network(self):
        functional.soft_update_network(self.q_network, self.target_q_network, self.target_smoothing_tau)
        functional.soft_update_network(self.policy_network, self.target_policy_network, self.target_smoothing_tau)
            
    def select_action(self, obs, deterministic=True):
        if type(obs) != torch.tensor:
            obs = torch.FloatTensor(np.array([obs])).to(util.device)
        action_info = self.policy_network.sample(obs)
        action = action_info['action_scaled']
        log_prob = action_info.get("log_prob", 1)
        return {
            'action': action.detach().cpu().numpy()[0],
            'log_prob': log_prob
            }


        



