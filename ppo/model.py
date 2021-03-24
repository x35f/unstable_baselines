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

class PPOAgent(torch.nn.Module, BaseAgent):
    def __init__(self,observation_space, action_space,
        gamma,
        normalize_advantage=True,
        clip_range=0.2, 
        **kwargs):
        state_dim = observation_space.shape[0]
        action_dim = action_space.shape[0]
        super(PPOAgent, self).__init__()
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
        self.v_optimizer = get_optimizer(kwargs['v_network']['optimizer_class'], self.v_network, kwargs['v_network']['learning_rate'])
        self.policy_optimizer = get_optimizer(kwargs['policy_network']['optimizer_class'], self.policy_network, kwargs['policy_network']['learning_rate'])

        #hyper-parameters
        self.gamma = gamma
        self.normalize_advantage = normalize_advantage
        self.clip_range = clip_range
        self.tot_update_count = 0 

    def update(self, data_batch):
        state_batch, action_batch, log_pi_batch, next_state_batch, reward_batch, future_return_batch, done_batch = data_batch
        curr_state_v = self.v_network(state_batch)
        next_state_v = self.v_network(next_state_batch)
        curr_log_pi = self.policy_network.log_prob(state_batch, action_batch)
        ratio_batch = torch.exp(curr_log_pi - log_pi_batch)
        
        #delta = reward_batch + self.gamma * (1 - done_batch) * next_state_v - curr_state_v
        advantages = future_return_batch - curr_state_v.detach()
        if self.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        
        surrogate1 = advantages * ratio_batch
        surrogate2 = torch.clamp(ratio_batch, 1 - self.clip_range, 1 + self.clip_range) * advantages
        min_surrogate = torch.min(surrogate1, surrogate2)

        #compute value loss
        v_loss = F.mse_loss(curr_state_v, future_return_batch)
        v_loss_value = v_loss.detach().cpu().numpy()
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()

        #compute policy loss=
        policy_loss = - min_surrogate.mean()
        policy_loss_value = policy_loss.detach().cpu().numpy()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.tot_update_count += 1
        
        return {
            "loss/v": v_loss_value, 
            "loss/policy": policy_loss_value,
        }
            
    def select_action(self, state, evaluate=False):
        if type(state) != torch.tensor:
            state = torch.FloatTensor([state]).to(util.device)
        action, log_prob, mean = self.policy_network.sample(state)
        if evaluate:
            return mean.detach().cpu().numpy()[0]
        else:
            return action.detach().cpu().numpy()[0]
    def act(self, state, evaluate=False):
        if type(state) != torch.tensor:
            state = torch.FloatTensor([state]).to(util.device)
        action, log_prob, mean = self.policy_network.sample(state)
        if evaluate:
            return mean.detach().cpu().numpy()[0]
        else:
            return action.detach().cpu().numpy()[0], log_prob

    def save_model(self, target_dir, ite):
        target_dir = os.path.join(target_dir, "ite_{}".format(ite))
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        #save q networks 
        save_path = os.path.join(target_dir, "V_network.pt")
        torch.save(self.v_network, save_path)
        #save policy network
        save_path = os.path.join(target_dir, "policy_network.pt")
        torch.save(self.policy_network, save_path)

    def load_model(self, model_dir):
        v_network_path = os.path.join(model_dir, "V_network.pt")
        self.v_network.load_state_dict(torch.load(v_network_path))
        policy_network_path = os.path.join(model_dir, "policy_network.pt")
        self.policy_network.load_state_dict(torch.load(policy_network_path))


        



