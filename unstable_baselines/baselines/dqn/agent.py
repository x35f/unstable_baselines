import torch
import torch.nn.functional as F
from unstable_baselines.common.buffer import ReplayBuffer
from unstable_baselines.common import util, functional
from unstable_baselines.common.agents import BaseAgent
from unstable_baselines.common.networks import SequentialNetwork, get_optimizer
import os


class DuelingQ(SequentialNetwork):
    def __init__(self, input_dim, out_dim, **kwargs):
        self.num_actions = out_dim - 1
        super(DuelingQ, self).__init__(input_dim, out_dim, **kwargs)

    def forward(self, input):
        x = self.networks(input)
        v = x[:,0].unsqueeze(1).repeat(1,self.num_actions)
        adv = x[:,1:]
        x = v + adv
        return x

    
class DQNAgent(BaseAgent):
    def __init__(self,
            observation_space,
            action_space,
            update_target_network_interval,     # C
            gamma,                        # reward discount
            target_smoothing_tau,
            **kwargs
            ):
        
        torch.nn.Module.__init__(self)
        BaseAgent.__init__(self, **kwargs)
        self.action_dim = action_space.n
        self.obs_shape = observation_space.shape
        self.double_q = kwargs['double']
        self.dueling = kwargs['dueling']
        self.is_visual_input = len(self.obs_shape) == 3
        #initilze networks
        #use v network for discrete action case
        if self.dueling:
            self.q_target_network = DuelingQ(self.obs_shape, self.action_dim+1,  **kwargs['q_network'])
            self.q_network = DuelingQ(self.obs_shape, self.action_dim+1, **kwargs['q_network'])
        else:
            self.q_target_network = SequentialNetwork(self.obs_shape, self.action_dim,  **kwargs['q_network'])
            self.q_network = SequentialNetwork(self.obs_shape, self.action_dim, **kwargs['q_network'])
        

        self.q_optimizer = get_optimizer(kwargs['q_network']['optimizer_class'], self.q_network, kwargs['q_network']['learning_rate'])
        
        #sync network
        functional.soft_update_network(self.q_network, self.q_target_network, 1.0)

        #pass to util.device
        self.q_target_network = self.q_target_network.to(util.device)
        self.q_network = self.q_network.to(util.device)

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
        if self.is_visual_input:
            obs_batch = obs_batch / 255.0
            next_obs_batch = next_obs_batch / 255.0
         #compute q_target
        with torch.no_grad():
            q_target_values = self.q_target_network(next_obs_batch)
            if self.double_q:
                best_action_idxs = self.q_network(next_obs_batch).max(1, keepdim=True)[1]
                q_target_values = self.q_target_network(next_obs_batch).gather(1, best_action_idxs).squeeze(-1)
            else:
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
        if self.is_visual_input:
            ob = ob / 255.0
        q_values = self.q_network(ob)
        q_values, action_indices = torch.max(q_values, dim=1)
        action = action_indices.detach().cpu().numpy()
        return {"action": action}