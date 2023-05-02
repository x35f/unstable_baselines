import torch
import torch.nn.functional as F
import random
from unstable_baselines.common.agents import BaseAgent
from unstable_baselines.common.networks import SequentialNetwork, PolicyNetworkFactory, get_optimizer
import numpy as np
from unstable_baselines.common import util, functional
from operator import itemgetter
import gym

class REDQAgent(BaseAgent):
    def __init__(self,observation_space, action_space,
            reward_scale,
            target_smoothing_tau,
            num_q_networks,
            num_q_samples,
            gamma,
            alpha,
            **kwargs):
        super(REDQAgent, self).__init__()
        obs_shape = observation_space.shape


        #initialize networks
        if isinstance(action_space, gym.spaces.discrete.Discrete):
            self.discrete_action_space = True
            action_dim = action_space.n
            self.q_networks = [SequentialNetwork(obs_shape, action_dim, ** kwargs['q_network']) for _ in range (num_q_networks)]
            self.q_target_networks = [SequentialNetwork(obs_shape, action_dim, ** kwargs['q_network']) for _ in range (num_q_networks)]
        elif isinstance(action_space, gym.spaces.box.Box):
            self.discrete_action_space = False
            action_dim = action_space.shape[0]
            self.q_networks = [SequentialNetwork(obs_shape[0] + action_dim, 1,** kwargs['q_network']) for _ in range (num_q_networks)]
            self.q_target_networks = [SequentialNetwork(obs_shape[0] + action_dim, 1,** kwargs['q_network']) for _ in range (num_q_networks)]
        else:
            assert 0, "unsupported action space for REDQ"
        
        self.policy_network = PolicyNetworkFactory.get(observation_space,action_space, ** kwargs['policy_network'])

        #sync q network parameters to target network
        for q_network_id in range(num_q_networks):
            functional.soft_update_network(self.q_networks[q_network_id], self.q_target_networks[q_network_id], 1.0)

        #pass to util.device
        for i in range(num_q_networks):
            self.q_networks[i] = self.q_networks[i].to(util.device)
            self.q_target_networks[i] = self.q_target_networks[i].to(util.device)
        self.policy_network = self.policy_network.to(util.device)
        
        #initialize optimizer
        self.q_optimizers = [get_optimizer(kwargs['q_network']['optimizer_class'], q_network, kwargs['q_network']['learning_rate']) for q_network in self.q_networks]
        self.policy_optimizer = get_optimizer(kwargs['policy_network']['optimizer_class'], self.policy_network, kwargs['policy_network']['learning_rate'])

        #entropy 
        self.automatic_entropy_tuning = kwargs['entropy']['automatic_tuning']
        self.alpha = alpha
        if self.automatic_entropy_tuning is True:
            if self.discrete_action_space:
                self.target_entropy = -np.log(1.0 / action_dim) * 0.98
            else:
                self.target_entropy = -np.prod(action_space.shape).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=util.device)
            self.alpha = self.log_alpha.detach().exp()
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=kwargs['entropy']['learning_rate'])
        
        #hyper-parameters
        self.gamma = gamma
        self.target_smoothing_tau = target_smoothing_tau
        self.reward_scale = reward_scale
        self.num_q_networks = num_q_networks
        self.num_q_samples = num_q_samples

    def update(self, data_batch, update_policy):
        if self.discrete_action_space:
            return self.discrete_action_space_update(data_batch, update_policy)
        else:
            return self.continuous_action_space_update(data_batch, update_policy)
        
    def discrete_action_space_update(self, data_batch, update_policy):
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch, truncated_batch = \
            itemgetter("obs", "action", "reward", "next_obs", "done", "truncated")(data_batch)
        
        reward_batch = reward_batch * self.reward_scale
        #update Q network for G times
        sampled_q_indices = random.sample(range(self.num_q_networks), self.num_q_samples)
        
        with torch.no_grad():
            next_obs_actions, next_obs_probs, next_obs_log_probs = itemgetter("action", "probs", "log_prob")(self.policy_network.sample(next_obs_batch))
            target_q_values = [self.q_target_networks[i](next_obs_batch) for i in sampled_q_indices]
            
            target_qs = torch.stack([torch.gather(target_q, 1, next_obs_actions) for target_q in target_q_values])
            
            min_target_q_value, _ = torch.min(target_qs, axis = 0)
            next_q = (next_obs_probs *(min_target_q_value - self.alpha * next_obs_log_probs)).sum(dim=1, keepdim=True)
            target_q = reward_batch + self.gamma * (1. - done_batch) * next_q

        curr_obs_q_values = [q_network(obs_batch) for q_network in self.q_networks]
        curr_obs_qs = [torch.gather(curr_obs_q_v, 1, action_batch.unsqueeze(1)) for curr_obs_q_v in curr_obs_q_values]

        q_loss_values = []

        for q_value, q_optim in zip(curr_obs_qs, self.q_optimizers):
            q_loss = F.mse_loss(q_value, target_q)
            q_loss_values.append(q_loss.item())
            q_optim.zero_grad()
            q_loss.backward()
            q_optim.step()
        self.update_target_network()
        
        if update_policy:
            #compute policy loss
            new_curr_actions, new_curr_action_probs, new_curr_action_log_probs = \
                itemgetter("action", "probs", "log_prob")(self.policy_network.sample(obs_batch))
            
            new_curr_obs_q_values =  [q_network(obs_batch) for q_network in self.q_networks]

            new_curr_obs_q_values = torch.stack(new_curr_obs_q_values)# (n, batch, action_dim)
            new_curr_min, _ = torch.min(new_curr_obs_q_values, axis=0)# (batch, action_dim)
            
            q = torch.sum(new_curr_min * new_curr_action_probs, dim=1, keepdim=True)
            entropies = - torch.sum(new_curr_action_probs  *new_curr_action_log_probs, dim=1, keepdim=True)
            
            policy_loss = (- q - self.alpha * entropies).mean()
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            #compute entropy loss
            if self.automatic_entropy_tuning:
                alpha_loss = -(self.log_alpha * (self.target_entropy - entropies.detach())).mean()
                alpha_loss_value = alpha_loss.item()
                self.alpha_optim.zero_grad()
                alpha_loss.backward()
                self.alpha_optim.step()

                self.alpha = self.log_alpha.exp()
                alpha_value = self.alpha.item()
            else:
                alpha_loss_value = 0
                alpha_value = self.alpha

        
            return {
                "loss/q_mean": np.mean(q_loss_values), 
                "loss/q_std": np.std(q_loss_values), 
                "loss/policy": policy_loss.item(),
                "loss/alpha": alpha_loss_value,
                "misc/alpha": alpha_value
            }
        else:
            return {
                "loss/q_mean": np.mean(q_loss_values), 
                "loss/q_std": np.std(q_loss_values)
            }
    
    def continuous_action_space_update(self, data_batch, update_policy):
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch, truncated_batch = \
            itemgetter("obs", "action", "reward", "next_obs", "done", "truncated")(data_batch)
        
        reward_batch = reward_batch * self.reward_scale
        #update Q network for G times
        sampled_q_indices = random.sample(range(self.num_q_networks), self.num_q_samples)
        
        with torch.no_grad():
            next_obs_actions, next_obs_log_prob = itemgetter("action", "log_prob")(self.policy_network.sample(next_obs_batch))
            target_q_values = torch.stack([self.q_target_networks[i](torch.cat([next_obs_batch, next_obs_actions], dim=-1)) for i in sampled_q_indices])
            min_target_q_value, _ = torch.min(target_q_values, axis = 0)
            q_target = reward_batch + self.gamma * (1. - done_batch) * (min_target_q_value - self.alpha * next_obs_log_prob)
        curr_obs_q_values = [q_network(torch.cat([obs_batch, action_batch], dim=1)) for q_network in self.q_networks]
        q_loss_values = []

        for q_value, q_optim in zip(curr_obs_q_values, self.q_optimizers):
            q_loss = F.mse_loss(q_value, q_target)
            q_loss_values.append(q_loss.item())
            q_optim.zero_grad()
            q_loss.backward()
            q_optim.step()
        self.update_target_network()
        
        if update_policy:
            #compute policy loss
            new_curr_obs_actions, new_curr_obs_log_pi = itemgetter("action", "log_prob")(self.policy_network.sample(obs_batch))
            new_curr_obs_q_values =  [q_network(torch.cat([obs_batch, new_curr_obs_actions], dim=-1)) for q_network in self.q_networks]
            new_curr_obs_q_values = torch.stack(new_curr_obs_q_values)
            new_mean_curr_obs_q_values = torch.mean(new_curr_obs_q_values, axis = 0)
            policy_loss = ((self.alpha * new_curr_obs_log_pi) - new_mean_curr_obs_q_values).mean()
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            #compute entropy loss
            if self.automatic_entropy_tuning:
                alpha_loss = -(self.log_alpha * (new_curr_obs_log_pi + self.target_entropy).detach()).mean()
                alpha_loss_value = alpha_loss.item()
                self.alpha_optim.zero_grad()
                alpha_loss.backward()
                self.alpha_optim.step()

                self.alpha = self.log_alpha.exp()
                alpha_value = self.alpha.item()
            else:
                alpha_loss_value = 0
                alpha_value = self.alpha

        
            return {
                "loss/q_mean": np.mean(q_loss_values), 
                "loss/q_std": np.std(q_loss_values), 
                "loss/policy": policy_loss.item(),
                "loss/alpha": alpha_loss_value,
                "misc/alpha": alpha_value
            }
        else:
            return {
                "loss/q_mean": np.mean(q_loss_values), 
                "loss/q_std": np.std(q_loss_values)
            }

    def update_target_network(self):
        for i in range(self.num_q_networks):
            functional.soft_update_network(self.q_networks[i], self.q_target_networks[i], self.target_smoothing_tau)
            
    @torch.no_grad()
    def select_action(self, obs, deterministic=False):
        if len(obs.shape) in [1, 3]:
            obs = [obs]
        if type(obs) != torch.tensor:
            obs = torch.FloatTensor(np.array(obs)).to(util.device)
        action, log_prob = itemgetter("action", "log_prob")(self.policy_network.sample(obs, deterministic=deterministic))
        if self.discrete_action_space:
            action = action[0]
        return {
            'action': action.detach().cpu().numpy(),
            'log_prob' : log_prob
            }

