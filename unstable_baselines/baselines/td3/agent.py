import torch
import torch.nn.functional as F
import gym 
import os
from torch import nn
from unstable_baselines.common.agents import BaseAgent
from unstable_baselines.common.networks import SequentialNetwork, PolicyNetworkFactory, get_optimizer
import numpy as np
from operator import itemgetter
from unstable_baselines.common import util, functional

class TD3Agent(BaseAgent):
    def __init__(self, 
        observation_space, 
        action_space,
        target_action_noise,
        noise_range,
        target_smoothing_tau,
        **kwargs
        ):
        super(TD3Agent, self).__init__()
        obs_dim = observation_space.shape[0]
        action_dim = action_space.shape[0]
        #save parameters
        self.args = kwargs

        #initilze networks
        self.q1_network = SequentialNetwork(obs_dim + action_dim, 1, **kwargs['q_network'])
        self.q2_network = SequentialNetwork(obs_dim + action_dim, 1, **kwargs['q_network'])
        self.target_q1_network = SequentialNetwork(obs_dim + action_dim, 1, **kwargs['q_network'])
        self.target_q2_network = SequentialNetwork(obs_dim + action_dim, 1, **kwargs['q_network'])
        self.policy_network = PolicyNetworkFactory.get(observation_space, action_space,  ** kwargs['policy_network'])
        self.target_policy_network = PolicyNetworkFactory.get(observation_space, action_space,  ** kwargs['policy_network'])

        #sync network parameters
        functional.soft_update_network(self.q1_network, self.target_q1_network, 1.0)
        functional.soft_update_network(self.q2_network, self.target_q2_network, 1.0)
        functional.soft_update_network(self.policy_network, self.target_policy_network, 1.0)

        #pass to util.util.device
        self.q1_network = self.q1_network.to(util.device)
        self.q2_network = self.q2_network.to(util.device)
        self.target_q1_network = self.target_q1_network.to(util.device)
        self.target_q2_network = self.target_q2_network.to(util.device)
        self.policy_network = self.policy_network.to(util.device)
        self.target_policy_network = self.target_policy_network.to(util.device)
        
        #initialize optimizer
        self.q1_optimizer = get_optimizer(kwargs['q_network']['optimizer_class'], self.q1_network, kwargs['q_network']['learning_rate'])
        self.q2_optimizer = get_optimizer(kwargs['q_network']['optimizer_class'], self.q2_network, kwargs['q_network']['learning_rate'])
        self.policy_optimizer = get_optimizer(kwargs['policy_network']['optimizer_class'], self.policy_network, kwargs['policy_network']['learning_rate'])

        #hyper-parameters
        self.gamma = kwargs['gamma']
        self.action_upper_bound = action_space.high[0]
        self.action_lower_bound = action_space.low[0]
        self.target_action_noise = target_action_noise
        self.noise_range = noise_range
        self.target_smoothing_tau = target_smoothing_tau


    def update(self, data_batch, update_policy_network):
        obs_batch,action_batch,next_obs_batch,reward_batch,done_batch = itemgetter('obs', 'action', 'next_obs', 'reward', 'done')(data_batch)
        
        curr_state_q1_value = self.q1_network(torch.cat([obs_batch, action_batch], dim=1))
        curr_state_q2_value = self.q2_network(torch.cat([obs_batch, action_batch], dim=1))
        

        with torch.no_grad():
            next_state_action_info = self.target_policy_network.sample(next_obs_batch)
            next_state_action = next_state_action_info['action_scaled']

            #apply noise to next state_action
            epsilon = torch.randn_like(next_state_action) * self.target_action_noise
            epsilon = torch.clamp(epsilon, -self.noise_range, self.noise_range)
            next_state_action = next_state_action + epsilon
            next_state_action = torch.clamp(next_state_action, self.action_lower_bound, self.action_upper_bound)

            #compute target value
            next_state_q1_value = self.target_q1_network(torch.cat([next_obs_batch, next_state_action], dim=1))
            next_state_q2_value = self.target_q2_network(torch.cat([next_obs_batch, next_state_action], dim=1))
            next_state_min_q = torch.min(next_state_q1_value, next_state_q2_value)
            target_q = reward_batch + self.gamma * (1. - done_batch) * next_state_min_q


        #compute q loss
        q1_loss = F.mse_loss(curr_state_q1_value, target_q)
        q2_loss = F.mse_loss(curr_state_q2_value, target_q)
        q_loss = q1_loss + q2_loss
        
        #update q functions
        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        q_loss.backward()
        self.q1_optimizer.step()
        self.q2_optimizer.step()

        log_info = {
                "loss/q1": q1_loss.item(), 
                "loss/q2": q2_loss.item(), 
                "loss/q": q_loss.item()
            }

        if update_policy_network:
            #compute policy loss
            new_curr_state_action_info = self.policy_network.sample(obs_batch)
            new_curr_state_action = new_curr_state_action_info['action_scaled']
            new_curr_state_q_value = self.q1_network(torch.cat([obs_batch, new_curr_state_action], dim=1))
            policy_loss = - new_curr_state_q_value.mean()

            #update policy network
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            #update target network
            self.update_target_network()

            log_info["loss/policy"] =  policy_loss.item()
        
        return log_info

        
    @torch.no_grad()
    def update_target_network(self):
            functional.soft_update_network(self.q1_network, self.target_q1_network, self.target_smoothing_tau)
            functional.soft_update_network(self.q2_network, self.target_q2_network, self.target_smoothing_tau)
            functional.soft_update_network(self.policy_network, self.target_policy_network, self.target_smoothing_tau)

    @torch.no_grad()
    def select_action(self, obs, deterministic=False):
        if len(obs.shape) == 1:
            ret_single = True
            obs = [obs]
        if type(obs) != torch.tensor:
            obs = torch.FloatTensor(np.array(obs)).to(util.device)
        action = itemgetter("action_scaled")(self.policy_network.sample(obs))
        log_prob = np.zeros([(action.shape[0]),])
        if ret_single:
            action = action[0]
            log_prob = log_prob[0]
        return {
            'action': action.detach().cpu().numpy(),
            'log_prob' : log_prob
            }




