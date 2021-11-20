import torch
import torch.nn.functional as F
import gym 
import os
from torch import nn
from common.agents import BaseAgent
from common.networks import QNetwork, VNetwork, PolicyNetwork, get_optimizer
from common.buffer import ReplayBuffer
import numpy as np
from common import util 

class DDPGAgent(torch.nn.Module, BaseAgent):
    def __init__(self,observation_space, action_space,
        update_target_network_interval=50, 
        target_smoothing_tau=0.1,
        alpha=0.2,
        **kwargs):
        state_dim = observation_space.shape[0]
        action_dim = action_space.shape[0]
        super(DDPGAgent, self).__init__()
        #save parameters
        self.args = kwargs

        # get per flag
        self.per = self.args.get('per', False)

        #initilze networks
        self.q_network = QNetwork(state_dim + action_dim, 1, **kwargs['q_network'])
        self.target_q_network = QNetwork(state_dim + action_dim, 1,**kwargs['q_network'])
        self.policy_network = PolicyNetwork(state_dim, action_space,  ** kwargs['policy_network'])
        self.target_policy_network = PolicyNetwork(state_dim, action_space,  ** kwargs['policy_network'])

        #sync network parameters
        util.soft_update_network(self.q_network, self.target_q_network, 1.0)
        util.soft_update_network(self.policy_network, self.target_policy_network, 1.0)

        #pass to util.device
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
        self.tot_update_count = 0 
        self.update_target_network_interval = update_target_network_interval
        self.target_smoothing_tau = target_smoothing_tau

    def update(self, data_batch):
        state_batch, action_batch, next_state_batch, reward_batch, done_batch = data_batch
        
        curr_state_q_value = self.q_network(state_batch, action_batch)
        
        new_curr_state_action, new_curr_state_log_pi, _ = self.policy_network.sample(state_batch)
        next_state_action, next_state_log_pi, _ = self.target_policy_network.sample(next_state_batch)

        new_curr_state_q_value = self.q_network(state_batch, new_curr_state_action)

        next_state_q_value = self.target_q_network(next_state_batch, next_state_action)
        target_q = reward_batch + self.gamma * (1. - done_batch) * next_state_q_value


        #compute q loss
        q_loss = F.mse_loss(curr_state_q_value, target_q.detach())

        q_loss_value = q_loss.detach().cpu().numpy()
        self.q_optimizer.zero_grad()
        q_loss.backward()

        #compute policy loss
        policy_loss = new_curr_state_q_value.mean()
        policy_loss_value = policy_loss.detach().cpu().numpy()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()

        self.q_optimizer.step()
        self.policy_optimizer.step()

        self.tot_update_count += 1

        
        #update target network
        self.try_update_target_network()

        return {
            "loss/q": q_loss_value, 
            "loss/policy": policy_loss_value, 
        }
        

    def try_update_target_network(self):
        if self.tot_update_count % self.update_target_network_interval == 0:
            util.soft_update_network(self.q_network, self.target_q_network, self.target_smoothing_tau)
            util.soft_update_network(self.policy_network, self.target_policy_network, self.target_smoothing_tau)
            
    def select_action(self, state, evaluate=False):
        if type(state) != torch.tensor:
            state = torch.FloatTensor(np.array([state])).to(util.device)
        action, log_prob, mean = self.policy_network.sample(state)
        if evaluate:
            return mean.detach().cpu().numpy()[0], log_prob
        else:
            return action.detach().cpu().numpy()[0], log_prob


        



