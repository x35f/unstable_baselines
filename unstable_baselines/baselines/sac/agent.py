import torch
import torch.nn.functional as F
from operator import itemgetter
from torch import nn
from unstable_baselines.common.agents import BaseAgent
# from unstable_baselines.common.networks import  MLPNetwork, PolicyNetwork, get_optimizer
from unstable_baselines.common.networks import SequentialNetwork, JointNetwork, PolicyNetworkFactory, get_optimizer
import numpy as np
import gym
from unstable_baselines.common import util, functional

class SACAgent(BaseAgent):
    def __init__(self,observation_space, action_space,
        target_smoothing_tau,
        alpha,
        reward_scale,
        **kwargs):
        super(SACAgent, self).__init__()
        
        obs_shape = observation_space.shape

        #initilize networks
        if isinstance(action_space, gym.spaces.discrete.Discrete):
            self.discrete_action_space = True
            action_dim = action_space.n
            self.q1_network = SequentialNetwork(obs_shape, action_dim, **kwargs['q_network'])
            self.q2_network = SequentialNetwork(obs_shape, action_dim, **kwargs['q_network'])
            self.target_q1_network = SequentialNetwork(obs_shape, action_dim, **kwargs['q_network'])
            self.target_q2_network = SequentialNetwork(obs_shape, action_dim, **kwargs['q_network'])
        elif isinstance(action_space, gym.spaces.box.Box):
            self.discrete_action_space = False
            action_dim = action_space.shape[0]
            if len(observation_space.shape) == 1:
                self.q1_network = SequentialNetwork(obs_shape[0] + action_dim, 1, **kwargs['q_network'])
                self.q2_network = SequentialNetwork(obs_shape[0] + action_dim, 1, **kwargs['q_network'])
                self.target_q1_network = SequentialNetwork(obs_shape[0] + action_dim, 1, **kwargs['q_network'])
                self.target_q2_network = SequentialNetwork(obs_shape[0] + action_dim, 1, **kwargs['q_network'])
            elif len(observation_space.shape) == 3:
                self.q1_network = JointNetwork([obs_shape, action_dim], kwargs['q_network']['embedding_sizes'], 1,  **kwargs['q_network'])
                self.q2_network = JointNetwork([obs_shape, action_dim], kwargs['q_network']['embedding_sizes'],1,  **kwargs['q_network'])
                self.target_q1_network = JointNetwork([obs_shape, action_dim], kwargs['policy_network']['embedding_sizes'], **kwargs['q_network'])
                self.target_q2_network = JointNetwork([obs_shape, action_dim], kwargs['policy_network']['embedding_sizes'], **kwargs['q_network'])
            else:
                assert 0, "unsopprted observation_space"
        else:
            assert 0, "unsupported action space for SAC"
        
        self.policy_network = PolicyNetworkFactory.get(observation_space, action_space, **kwargs["policy_network"])

        #sync network parameters
        functional.soft_update_network(self.q1_network, self.target_q1_network, 1.0)
        functional.soft_update_network(self.q2_network, self.target_q2_network, 1.0)

        #pass to util.device
        self.q1_network = self.q1_network.to(util.device)
        self.q2_network = self.q2_network.to(util.device)
        self.target_q1_network = self.target_q1_network.to(util.device)
        self.target_q2_network = self.target_q2_network.to(util.device)
        self.policy_network = self.policy_network.to(util.device)

        #initialize optimizer
        self.q1_optimizer = get_optimizer(kwargs['q_network']['optimizer_class'], self.q1_network, kwargs['q_network']['learning_rate'])
        self.q2_optimizer = get_optimizer(kwargs['q_network']['optimizer_class'], self.q2_network, kwargs['q_network']['learning_rate'])
        self.policy_optimizer = get_optimizer(kwargs['policy_network']['optimizer_class'], self.policy_network, kwargs['policy_network']['learning_rate'])

        #entropy
        self.automatic_entropy_tuning = kwargs['entropy']['automatic_tuning']

        self.alpha = alpha
        if self.automatic_entropy_tuning:
            if self.discrete_action_space:
                self.target_entropy = - np.log(1.0 / action_dim) *  kwargs['entropy']['scale']
                # self.target_entropy = - action_dim
            else:
                self.target_entropy = -np.prod(action_space.shape).item()
            self.log_alpha = torch.zeros(1,  requires_grad=True, device=util.device)
            self.alpha = self.log_alpha.detach().exp()
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=kwargs['entropy']['learning_rate'])

        #hyper-parameters
        self.gamma = kwargs['gamma']
        self.target_smoothing_tau = target_smoothing_tau
        self.reward_scale = reward_scale


    def update(self, data_batch):
        if self.discrete_action_space:
            return self.discrete_action_space_update(data_batch)
        else:
            return self.continuous_action_space_update(data_batch)

    def discrete_action_space_update(self, data_batch):
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch, truncated_batch = \
            itemgetter("obs", "action", "reward", "next_obs", "done", "truncated")(data_batch)
        
        reward_batch = reward_batch * self.reward_scale

        #calculate critic loss
        curr_obs_q1_values = self.q1_network(obs_batch)
        curr_q1 = torch.gather(curr_obs_q1_values, 1, action_batch.unsqueeze(1))
        curr_obs_q2_values = self.q2_network(obs_batch)
        curr_q2 = torch.gather(curr_obs_q2_values, 1, action_batch.unsqueeze(1))
        
        with torch.no_grad():
            next_obs_actions, next_obs_action_probs, next_obs_log_probs = \
                itemgetter("action", "probs", "log_prob")(self.policy_network.sample(next_obs_batch))
            target_q1_values = self.target_q1_network(next_obs_batch)
            target_q1 = torch.gather(target_q1_values, 1, next_obs_actions)
            target_q2_values = self.target_q2_network(next_obs_batch)
            target_q2 = torch.gather(target_q2_values, 1, next_obs_actions)
            target_q = (next_obs_action_probs *(torch.min(target_q1, target_q2) - self.alpha * next_obs_log_probs)).sum(dim=1, keepdim=True)
            target_q = reward_batch + self.gamma * (1. - done_batch) * target_q

        q1_loss = F.mse_loss(curr_q1, target_q)
        q2_loss = F.mse_loss(curr_q2, target_q)

        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        (q1_loss + q2_loss).backward()
        self.q1_optimizer.step()
        self.q2_optimizer.step()

        #calculate actor loss
        new_curr_actions, new_curr_action_probs, new_curr_action_log_probs = \
            itemgetter("action", "probs", "log_prob")(self.policy_network.sample(obs_batch))
        curr_q1_values = self.q1_network(obs_batch)
        curr_q2_values = self.q2_network(obs_batch)

        entropies = - torch.sum(new_curr_action_probs *new_curr_action_log_probs, dim=1, keepdim=True)
        q = torch.sum(torch.min(curr_q1_values, curr_q2_values) * new_curr_action_probs, dim=1, keepdim=True)
       
        policy_loss = (- q - self.alpha * entropies).mean()
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (self.target_entropy - entropies.detach())).mean()
            alpha_loss_value = alpha_loss.detach().cpu().item()
            self.alpha_optim.zero_grad()
        else:
            alpha_loss = 0.
            alpha_loss_value = 0.   
        #update actor and entropy
        self.policy_optimizer.zero_grad()
        (policy_loss + alpha_loss).backward()
        self.policy_optimizer.step()
        if self.automatic_entropy_tuning:
            self.alpha_optim.step()
            self.alpha = self.log_alpha.detach().exp()

        self.update_target_network()
        return {
            "loss/q1": q1_loss.item(), 
            "loss/q2": q2_loss.item(), 
            "loss/policy": policy_loss.item(), 
            "loss/entropy": alpha_loss_value, 
            "misc/entropy_alpha": self.alpha.item(),
            "misc/entropy": entropies.mean().item(),
        }

    
    def continuous_action_space_update(self, data_batch):
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch, truncated_batch = \
            itemgetter("obs", "action", "reward", "next_obs", "done", "truncated")(data_batch)
        
       
        reward_batch = reward_batch * self.reward_scale
        
        #calculate critic loss
        curr_q1 = self.q1_network([obs_batch, action_batch])
        curr_q2 = self.q2_network([obs_batch, action_batch])
        with torch.no_grad():
            next_obs_action, next_obs_log_prob = \
                itemgetter("action", "log_prob")(self.policy_network.sample(next_obs_batch))

            target_q1_value = self.target_q1_network([next_obs_batch, next_obs_action])
            target_q2_value = self.target_q2_network([next_obs_batch, next_obs_action])
            next_obs_min_q = torch.min(target_q1_value, target_q2_value)
            target_q = (next_obs_min_q - self.alpha * next_obs_log_prob)
            target_q = reward_batch + self.gamma * (1. - done_batch) * target_q

        #compute q loss and backward
        q1_loss = F.mse_loss(curr_q1, target_q)
        q2_loss = F.mse_loss(curr_q2, target_q)

        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        (q1_loss + q2_loss).backward()
        self.q1_optimizer.step()
        self.q2_optimizer.step()

        ##########

        new_curr_obs_action, new_curr_obs_log_prob = \
            itemgetter("action", "log_prob")(self.policy_network.sample(obs_batch))
        new_curr_obs_q1_value = self.q1_network([obs_batch, new_curr_obs_action])
        new_curr_obs_q2_value = self.q2_network([obs_batch, new_curr_obs_action])
        new_min_curr_obs_q_value = torch.min(new_curr_obs_q1_value, new_curr_obs_q2_value)
        #print(new_curr_obs_log_prob.shape, self.target_entropy, new_curr_obs_log_prob.mean())
        #compute policy and ent loss
        policy_loss = ((self.alpha * new_curr_obs_log_prob) - new_min_curr_obs_q_value).mean()
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (new_curr_obs_log_prob + self.target_entropy).detach()).mean()
            alpha_loss_value = alpha_loss.detach().cpu().item()
            self.alpha_optim.zero_grad()
        else:
            alpha_loss = 0.
            alpha_loss_value = 0.
        
        self.policy_optimizer.zero_grad()
        (policy_loss + alpha_loss).backward()
        self.policy_optimizer.step()
        if self.automatic_entropy_tuning:
            self.alpha_optim.step()
            self.alpha = self.log_alpha.detach().exp()

        self.update_target_network()
        
        return {
            "loss/q1": q1_loss.item(), 
            "loss/q2": q2_loss.item(), 
            "loss/policy": policy_loss.item(), 
            "loss/entropy": alpha_loss_value, 
            "misc/entropy_alpha": self.alpha.item(),
        }
        

    def update_target_network(self):
        functional.soft_update_network(self.q1_network, self.target_q1_network, self.target_smoothing_tau)
        functional.soft_update_network(self.q2_network, self.target_q2_network, self.target_smoothing_tau)
            
    @torch.no_grad()
    def select_action(self, obs, deterministic=False):
        if len(obs.shape) in [1,3]:
            obs = [obs]
        if type(obs) != torch.tensor:
            obs = torch.FloatTensor(np.array(obs)).to(util.device)
        action, log_prob = itemgetter("action", "log_prob")(self.policy_network.sample(obs, deterministic=deterministic))
        if self.discrete_action_space:
            action = action[0]
        return {
            'action': action.detach().cpu().numpy(),
            'log_prob' : log_prob[0]
            }
