import torch
import torch.nn.functional as F
from common.buffer import ReplayBuffer
from common import util
from common.agents import BaseAgent
from common.agents import VNetwork, get_optimizer
import os

class DQNAgent(torch.nn.Module, BaseAgent):
    def __init__(self,
            observation_space,
            action_space,
            update_target_network_interval = 100,     # C
            gamma = 0.9,                        # reward discount
            tau = 0.5,                          # soft update parameter 
            n = 1,
            **kwargs
            ):
        
        super(DQNAgent, self).__init__()
        self.action_dim = action_space.n
        self.obs_dim = observation_space.shape[0]

        #initilze networks
        #use v network for discrete action case
        self.q_target_network = VNetwork(self.obs_dim, self.action_dim,  **kwargs['q_network'])
        self.q_network = VNetwork(self.obs_dim, self.action_dim, **kwargs['q_network'])
        #initialize optimizer
        self.q_optimizer = get_optimizer(kwargs['q_network']['optimizer_class'], self.q_network, kwargs['q_network']['learning_rate'])
        
        #sync network
        util.hard_update_network(self.q_network, self.q_target_network)

        #pass to util.device
        self.q_target_network = self.q_target_network.to(util.device)
        self.q_network = self.q_network.to(util.device)

        #hyper-parameters
        self.gamma = gamma
        self.tau = tau
        self.update_target_network_interval = update_target_network_interval
        self.tot_num_updates = 0
        self.n = n

    def update(self, data_batch):
        state_batch, action_batch, next_state_batch, reward_batch, done_batch = data_batch
         #compute q_target
        with torch.no_grad():
            q_target_values = self.q_target_network(next_state_batch)
            q_target_values, q_target_actions = torch.max(q_target_values, dim =1)
            q_target_values = q_target_values.unsqueeze(1)
            q_target = reward_batch + (1. - done_batch) * (self.gamma ** self.n) * q_target_values
        
        #compute q current
        q_current_values = self.q_network(state_batch)
        #q_current = torch.stack([_[idx] for _, idx in zip(q_current_values, action_batch)])
        q_current = torch.gather(q_current_values, 1, action_batch)
        #compute loss
        loss = F.mse_loss(q_target, q_current)
        loss_val = loss.detach().cpu().numpy()
        self.q_optimizer.zero_grad()
        loss.backward()
        self.q_optimizer.step()
        self.tot_num_updates += 1

        return {
            "loss/mse": loss_val
        }


    def try_update_target_network(self):
        if self.tot_num_updates % self.update_target_network_interval == 0:
            util.soft_update_network(self.q_network, self.q_target_network, self.tau)   

    def select_action(self, obs):
        ob = torch.tensor(obs).to(util.device).unsqueeze(0).float()
        q_values = self.q_network(ob)
        q_values, action_indices = torch.max(q_values, dim=1)
        action = action_indices.detach().cpu().numpy()[0]
        return action

    def save_model(self, target_dir, ite):
        target_dir = os.path.join(target_dir, "ite_{}".format(ite))
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        #save q networks
        save_path = os.path.join(target_dir, "q_network.pt")
        torch.save(self.q_network, save_path)

    def load_model(self, model_dir):
        q_network_path = os.path.join(model_dir, "q_network.pt")
        self.q_network.load_state_dict(torch.load(q_network_path))

