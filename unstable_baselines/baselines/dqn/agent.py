import torch
import torch.nn.functional as F
from unstable_baselines.common.buffer import ReplayBuffer
from unstable_baselines.common import util, functional
from unstable_baselines.common.agents import BaseAgent
from unstable_baselines.common.networks import MLPNetwork, get_optimizer
import os

class DQNAgent(torch.nn.Module, BaseAgent):
    def __init__(self,
            observation_space,
            action_space,
            update_target_network_interval,     # C
            gamma,                        # reward discount
            target_smoothing_tau,
            **kwargs
            ):
        
        super(DQNAgent, self).__init__()
        self.action_dim = action_space.n
        self.obs_dim = observation_space.shape[0]

        #initilze networks
        #use v network for discrete action case
        self.q_target_network = MLPNetwork(self.obs_dim, self.action_dim,  **kwargs['q_network'])
        self.q_network = MLPNetwork(self.obs_dim, self.action_dim, **kwargs['q_network'])
        #initialize optimizer
        self.q_optimizer = get_optimizer(kwargs['q_network']['optimizer_class'], self.q_network, kwargs['q_network']['learning_rate'])
        
        #sync network
        functional.soft_update_network(self.q_network, self.q_target_network, 1.0)

        #pass to util.device
        self.q_target_network = self.q_target_network.to(util.device)
        self.q_network = self.q_network.to(util.device)

        #register networks
        self.networks = {
            'q_network': self.q_network,
            'q_target_network': self.q_target_network
        }

        #hyper-parameters
        self.gamma = gamma
        self.target_smoothing_tau = target_smoothing_tau
        self.update_target_network_interval = update_target_network_interval
        self.tot_num_updates = 0

    def update(self, data_batch):
        obs_batch = data_batch['obs']
        action_batch = data_batch['action'] 
        next_obs_batch = data_batch['next_obs'] 
        reward_batch = data_batch['reward']
        done_batch = data_batch['done']
         #compute q_target
        with torch.no_grad():
            q_target_values = self.q_target_network(next_obs_batch)
            q_target_values, q_target_actions = torch.max(q_target_values, dim =1)
            q_target_values = q_target_values.unsqueeze(1)
            q_target = reward_batch + (1. - done_batch) * self.gamma * q_target_values
        
        #compute q current
        q_current_values = self.q_network(obs_batch)
        #q_current = torch.stack([_[idx] for _, idx in zip(q_current_values, action_batch)])
        q_current = torch.gather(q_current_values, 1, action_batch.unsqueeze(1))
        #compute loss
        loss = F.mse_loss(q_target, q_current)
        loss_val = loss.detach().cpu().numpy()
        self.q_optimizer.zero_grad()
        loss.backward()
        self.q_optimizer.step()
        self.tot_num_updates += 1

        self.update_target_network()
        
        return {
            "loss/mse": loss_val
        }

    def update_target_network(self):
        functional.soft_update_network(self.q_network, self.q_target_network, self.target_smoothing_tau)   

    def select_action(self, obs, deterministic=False):
        ob = torch.tensor(obs).to(util.device).unsqueeze(0).float()
        q_values = self.q_network(ob)
        q_values, action_indices = torch.max(q_values, dim=1)
        action = action_indices.detach().cpu().numpy()[0]
        return {"action": action}