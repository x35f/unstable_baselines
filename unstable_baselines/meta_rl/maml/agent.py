import torch
import torch.nn.functional as F
import gym 
import os
from torch import nn
from unstable_baselines.common.agents import BaseAgent
from unstable_baselines.common.networks import MLPNetwork, PolicyNetwork, get_optimizer
from unstable_baselines.common.buffer import ReplayBuffer
import numpy as np
from unstable_baselines.common import util 

class SACAgent(torch.nn.Module, BaseAgent):
    def __init__(self,observation_space, action_space,
        update_target_network_interval=50, 
        target_smoothing_tau=0.1,
        alpha=0.2,
        **kwargs):
        obs_dim = observation_space.shape[0]
        action_dim = action_space.shape[0]
        super(SACAgent, self).__init__()
        #save parameters
        self.args = kwargs

        #initilze networks
        self.q1_network = MLPNetwork(obs_dim + action_dim, 1, **kwargs['q_network'])
        self.q2_network = MLPNetwork(obs_dim + action_dim, 1,**kwargs['q_network'])
        self.target_q1_network = MLPNetwork(obs_dim + action_dim, 1,**kwargs['q_network'])
        self.target_q2_network = MLPNetwork(obs_dim + action_dim, 1,**kwargs['q_network'])
        self.policy_network = PolicyNetwork(obs_dim, action_space,  ** kwargs['policy_network'])

        #sync network parameters
        util.soft_update_network(self.q1_network, self.target_q1_network, 1.0)
        util.soft_update_network(self.q2_network, self.target_q2_network, 1.0)

        #pass to util.device
        self.q1_network = self.q1_network.to(util.device)
        self.q2_network = self.q2_network.to(util.device)
        self.target_q1_network = self.target_q1_network.to(util.device)
        self.target_q2_network = self.target_q2_network.to(util.device)
        self.policy_network = self.policy_network.to(util.device)

        #register networks
        self.networks = {
            'q1_network': self.q1_network,
            'q2_network': self.q2_network,
            'target_q1_network': self.target_q1_network,
            'target_q2_network': self.target_q2_network,
            'policy_network': self.policy_network
        }

        #initialize optimizer
        self.q1_optimizer = get_optimizer(kwargs['q_network']['optimizer_class'], self.q1_network, kwargs['q_network']['learning_rate'])
        self.q2_optimizer = get_optimizer(kwargs['q_network']['optimizer_class'], self.q2_network, kwargs['q_network']['learning_rate'])
        self.policy_optimizer = get_optimizer(kwargs['policy_network']['optimizer_class'], self.policy_network, kwargs['policy_network']['learning_rate'])

        #hyper-parameters
        self.gamma = kwargs['gamma']
        self.automatic_entropy_tuning = kwargs['entropy']['automatic_tuning']
        self.alpha = alpha
        if self.automatic_entropy_tuning is True:
            self.target_entropy = -np.prod(action_space.shape).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=util.device)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=kwargs['entropy']['learning_rate'])
        self.tot_update_count = 0 
        self.update_target_network_interval = update_target_network_interval
        self.target_smoothing_tau = target_smoothing_tau

    def update(self, data_batch):
        obs_batch, action_batch, next_obs_batch, reward_batch, done_batch = data_batch
        
        curr_state_q1_value = self.q1_network(torch.cat([obs_batch, action_batch],dim=1))
        curr_state_q2_value = self.q2_network(torch.cat([obs_batch, action_batch],dim=1))
        new_curr_state_action, new_curr_state_log_pi, _, _ = self.policy_network.sample(obs_batch)
        next_state_action, next_state_log_pi, _, _ = self.policy_network.sample(next_obs_batch)

        new_curr_state_q1_value = self.q1_network(torch.cat([obs_batch, new_curr_state_action],dim=1))
        new_curr_state_q2_value = self.q2_network(torch.cat([obs_batch, new_curr_state_action],dim=1))

        next_state_q1_value = self.target_q1_network(torch.cat([next_obs_batch, next_state_action],dim=1))
        next_state_q2_value = self.target_q2_network(torch.cat([next_obs_batch, next_state_action],dim=1))
        next_state_min_q = torch.min(next_state_q1_value, next_state_q2_value)
        target_q = (next_state_min_q - self.alpha * next_state_log_pi)
        target_q = reward_batch + self.gamma * (1. - done_batch) * target_q

        new_min_curr_state_q_value = torch.min(new_curr_state_q1_value, new_curr_state_q2_value)

        #compute q loss
        
        q1_loss = F.mse_loss(curr_state_q1_value, target_q.detach())
        q2_loss = F.mse_loss(curr_state_q2_value, target_q.detach())

        q1_loss_value = q1_loss.detach().cpu().numpy()
        q2_loss_value = q2_loss.detach().cpu().numpy()
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q2_optimizer.zero_grad()
        q2_loss.backward()

        #compute policy loss
        policy_loss = ((self.alpha * new_curr_state_log_pi) - new_min_curr_state_q_value).mean()
        policy_loss_value = policy_loss.detach().cpu().numpy()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()

        #compute entropy loss
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (new_curr_state_log_pi + self.target_entropy).detach()).mean()
            alpha_loss_value = alpha_loss.detach().cpu().numpy()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_value = self.alpha.detach().cpu().numpy()
        else:
            alpha_loss_value = 0.
            alpha_value = self.alpha
        self.tot_update_count += 1

        self.q1_optimizer.step()
        self.q2_optimizer.step()
        self.policy_optimizer.step()

        self.try_update_target_network()
        
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
            
    def select_action(self, state, deterministic=False):
        if type(state) != torch.tensor:
            state = torch.FloatTensor(np.array([state])).to(util.device)
        action, log_prob, mean, std = self.policy_network.sample(state)
        if deterministic:
            return mean.detach().cpu().numpy()[0], log_prob
        else:
            return action.detach().cpu().numpy()[0], log_prob
