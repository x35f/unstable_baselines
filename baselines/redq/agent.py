import torch
import torch.nn.functional as F
import gym 
import os
import random
from torch import nn
from common.agents import BaseAgent
from common.networks import MLPNetwork, PolicyNetwork, get_optimizer
from common.buffer import ReplayBuffer
import numpy as np
from common import util 

class REDQAgent(torch.nn.Module, BaseAgent):
    def __init__(self,observation_space, action_space,
            update_target_network_interval = 50, 
            target_smoothing_tau = 0.005,
            num_q_networks = 10,
            num_q_samples = 2,
            num_updates_per_iteration=20,
            automatic_entropy_tuning=False,
            gamma = 0.99,
            alpha=0.2,
            **kwargs):
        obs_dim = observation_space.shape[0]
        action_dim = action_space.shape[0]
        super(REDQAgent, self).__init__()
        #save parameters
        self.args = kwargs
        #initilze networks
        self.q_networks = [MLPNetwork(obs_dim + action_dim, 1,** kwargs['q_network']) for _ in range (num_q_networks)]
        self.q_target_networks = [MLPNetwork(obs_dim + action_dim, 1,** kwargs['q_network']) for _ in range (num_q_networks)]
        self.policy_network = PolicyNetwork(obs_dim,action_space, ** kwargs['policy_network'])

        #sync q network parameters to target network
        for q_network_id in range(num_q_networks):
            util.soft_update_network(self.q_networks[q_network_id], self.q_target_networks[q_network_id], 1.0)

        #pass to util.device
        for i in range(num_q_networks):
            self.q_networks[i] = self.q_networks[i].to(util.device)
            self.q_target_networks[i] = self.q_target_networks[i].to(util.device)
        self.policy_network = self.policy_network.to(util.device)

        #register networks
        self.networks = {}
        for i in range(num_q_networks):
            self.networks['q_network_{}'.format(i)] = self.q_networks[i]
        self.networks['policy_network'] = self.policy_network
        
        #initialize optimizer
        self.q_optimizers = [get_optimizer(kwargs['q_network']['optimizer_class'], q_network, kwargs['q_network']['learning_rate']) for q_network in self.q_networks]
        self.policy_optimizer = get_optimizer(kwargs['policy_network']['optimizer_class'], self.policy_network, kwargs['policy_network']['learning_rate'])

        #hyper-parameters
        self.gamma = gamma
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.alpha = alpha
        if self.automatic_entropy_tuning is True:
            self.target_entropy = -np.prod(action_space.shape).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=util.device)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=kwargs['entropy']['learning_rate'])
        self.tot_update_count = 0 
        self.update_target_network_interval = update_target_network_interval
        self.target_smoothing_tau = target_smoothing_tau
        self.num_q_networks = num_q_networks
        self.num_q_samples = num_q_samples
        self.num_updates_per_iteration = num_updates_per_iteration

    def update(self, data_batch):
        obs_batch, action_batch, next_obs_batch, reward_batch, done_batch = data_batch
        for update in range(self.num_updates_per_iteration): 
            #update Q network for G times
            sampled_q_indices = random.sample(range(self.num_q_networks), self.num_q_samples)
            next_state_actions, next_state_log_pi, _ = self.policy_network.sample(next_obs_batch)
            target_q_values = torch.stack([self.q_target_networks[i](next_obs_batch, next_state_actions) for i in sampled_q_indices])
            min_target_q_value, _ = torch.min(target_q_values, axis = 0)
            q_target = reward_batch + self.gamma * (1. - done_batch) * (min_target_q_value - self.alpha * next_state_log_pi)
            q_target = q_target.detach()
            curr_state_q_values = [q_network(obs_batch, action_batch) for q_network in self.q_networks]
            q_loss_values = []
            for q_value, q_optim in zip(curr_state_q_values, self.q_optimizers):
                q_loss = F.mse_loss(q_value, q_target)
                q_loss_value = q_loss.detach().cpu().numpy()
                q_loss_values.append(q_loss_value)
                q_optim.zero_grad()
                q_loss.backward()
                q_optim.step()

        #compute policy loss
        new_curr_state_actions, new_curr_state_log_pi, _ = self.policy_network.sample(obs_batch)
        new_curr_state_q_values =  [q_network(obs_batch, new_curr_state_actions) for q_network in self.q_networks]
        new_curr_state_q_values = torch.stack(new_curr_state_q_values)
        new_mean_curr_state_q_values = torch.mean(new_curr_state_q_values, axis = 0)
        policy_loss = ((self.alpha * new_curr_state_log_pi) - new_mean_curr_state_q_values).mean()
        policy_loss_value = policy_loss.detach().cpu().numpy()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

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
            alpha_loss_value = 0
            alpha_value = self.alpha
        self.tot_update_count += 1


        self.try_update_target_network()
        
        return {
            "loss/q_mean": np.mean(q_loss_values), 
            "loss/q_std": np.std(q_loss_values), 
            "loss/policy": policy_loss_value,
            "loss/alpha": alpha_loss_value,
            "info/alpha": alpha_value
        }

    def try_update_target_network(self):
        if self.tot_update_count % self.update_target_network_interval == 0:
            for i in range(self.num_q_networks):
                util.soft_update_network(self.q_networks[i], self.q_target_networks[i], self.target_smoothing_tau)
            
    def select_action(self, state, evaluate=False):
        if type(state) != torch.tensor:
            state = torch.FloatTensor(np.array([state])).to(util.device)
        action, log_prob, mean = self.policy_network.sample(state)
        if evaluate:
            return mean.detach().cpu().numpy()[0]
        else:
            return action.detach().cpu().numpy()[0]
