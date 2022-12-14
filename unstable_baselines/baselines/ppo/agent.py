from operator import itemgetter
import torch
import torch.nn.functional as F
import os
from torch import nn
from unstable_baselines.common.agents import BaseAgent
from unstable_baselines.common.networks import SequentialNetwork, PolicyNetworkFactory, get_optimizer
import numpy as np
from unstable_baselines.common import util 
import math

class PPOAgent(BaseAgent):
    def __init__(self,observation_space, action_space,
            beta: float,
            advantage_type: str,
            gamma: 0.99,
            advantage_params: 0.97,
            normalize_advantage:True,
            policy_loss_type: str, #"clipped_surrogate"
            entropy_coeff: float,
            c1: float,
            c2: float,
            clip_range: float, 
            adaptive_kl_coeff: bool,
            train_policy_iters: int,
            train_v_iters: int,
            target_kl: float,
            **kwargs):
        super(PPOAgent, self).__init__()
        assert policy_loss_type in ['naive', 'clipped_surrogate','adaptive_kl']
        
        #initilze networks
        self.v_network = SequentialNetwork(observation_space.shape, 1, **kwargs['v_network'])
        # print(observation_space, action_space)
        # exit(0)
        self.policy_network = PolicyNetworkFactory.get(observation_space, action_space,  **kwargs['policy_network'])

        #pass to util.device
        self.v_network = self.v_network.to(util.device)
        self.policy_network = self.policy_network.to(util.device)

        #initialize optimizer
        self.v_optimizer = get_optimizer(kwargs['v_network']['optimizer_class'], self.v_network, kwargs['v_network']['learning_rate'])
        self.policy_optimizer = get_optimizer(kwargs['policy_network']['optimizer_class'], self.policy_network, kwargs['policy_network']['learning_rate'])

        #hyper-parameters

        #advantage related parameters
        self.advantage_type = advantage_type
        self.gamma = gamma
        self.normalize_advantage = normalize_advantage
        self.advantage_params = advantage_params

        #policy loss related hyper-parameters
        self.policy_loss_type = policy_loss_type

        #entropy related
        self.entropy_coeff = entropy_coeff
        self.c1 = c1
        self.c2 = c2
        
        #adaptive kl coefficient related parameters
        self.adaptive_kl_coeff = adaptive_kl_coeff
        self.beta = beta
        self.target_kl = target_kl

        #clipping related hyper-parameters
        self.clip_range = clip_range

        #update counts
        self.train_v_iters = train_v_iters
        self.train_policy_iters = train_policy_iters

    @torch.no_grad()
    def estimate_advantage(self, obs_batch, action_batch, reward_batch, next_obs_batch, done_batch, truncated_batch):
        if self.advantage_type == "gae":
            gae_lambda = self.advantage_params['lambda']
            value_batch = self.v_network(obs_batch).data
            next_value_batch = self.v_network(next_obs_batch).data
            advantage_batch = torch.FloatTensor(action_batch.shape[0], 1).to(util.device)
            return_batch = torch.FloatTensor(action_batch.shape[0], 1).to(util.device)
            # terminal_batch = np.logical_or(done_batch.cpu().numpy(), truncated_batch.cpu().numpy())
            # terminal_batch = torch.FloatTensor(terminal_batch).to(util.device)
            # print(terminal_batch)
            # print(1.0-terminal_batch)
            #print(terminal_batch)
            #exit(0)
            #terminal_batch = done_batch
            delta_batch = reward_batch + next_value_batch * self.gamma - value_batch
            discount_batch = (1.0 - done_batch) * (1.0 - truncated_batch) * self.gamma * gae_lambda
            gae = 0.0
            for i in reversed(range(reward_batch.size(0))):
                # if truncated_batch[i] or done_batch[i]:
                #     gae = 0
                gae = delta_batch[i] + discount_batch[i] * gae
                advantage_batch[i] = gae
            return_batch = advantage_batch + value_batch


            # return_batch = torch.FloatTensor(action_batch.shape[0], 1).to(util.device)
            # deltas = torch.FloatTensor(action_batch.shape[0], 1).to(util.device)
            # advantage_batch = torch.FloatTensor(action_batch.shape[0], 1).to(util.device)
            # prev_return = 0
            # prev_value = 0
            # prev_advantage = 0
            # for i in reversed(range(reward_batch.size(0))):
            #     if truncated_batch[i] and not done_batch[i]:
            #         prev_value = value_batch[i]
            #     elif done_batch[i]:
            #         prev_value = 0
            #     return_batch[i] = reward_batch[i] + self.gamma * prev_return * (1.0 - done_batch[i])
            #     deltas[i] = reward_batch[i] + self.gamma * prev_value * (1.0 - done_batch[i]) - value_batch.data[i]
            #     advantage_batch[i] = deltas[i] + self.gamma * gae_lambda * prev_advantage * (1.0 - done_batch[i])

            #     prev_return = return_batch[i, 0]
            #     prev_value = value_batch.data[i, 0]
            #     prev_advantage = advantage_batch[i, 0]
        else:
            raise NotImplementedError

        #normalize advantage
        if self.normalize_advantage:
            advantage_batch =  (advantage_batch - advantage_batch.mean()) / (advantage_batch.std() + 1e-8)

        return advantage_batch, return_batch

    def update(self, data_batch):
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch, truncated_batch = \
            itemgetter("obs", "action", "reward", "next_obs", "done", "truncated")(data_batch)


        #obs_batch, action_batch, log_prob_batch,  advantage_batch, return_batch = \
        #    itemgetter("obs", "action", "log_prob", "advantage", "ret")(data_batch)
        
        #compute log_prob, advantage, return from data batch
        
        advantage_batch, return_batch =  self.estimate_advantage(obs_batch, action_batch, reward_batch, next_obs_batch, done_batch, truncated_batch)
        with torch.no_grad():
            log_prob_batch = itemgetter("log_prob")(self.policy_network.evaluate_actions(obs_batch, action_batch)).data
     
        #update policy
        update_policy_counts = 0
        for update_policy_step in range(self.train_policy_iters): 
            #compute and step policy loss
            new_log_prob, dist_entropy = itemgetter("log_prob", "entropy")(self.policy_network.evaluate_actions(obs_batch, action_batch))
            
            ratio_batch = torch.exp(new_log_prob - log_prob_batch)
            approx_kl = (log_prob_batch - new_log_prob).mean().item()
            if approx_kl > 1.5 * self.target_kl:
                break
            if self.policy_loss_type == "clipped_surrogate":
                surrogate1 = advantage_batch * ratio_batch
                #print(self.clip_range, advantages.shape, ratio_batch.shape)
                surrogate2 =  advantage_batch * torch.clamp(ratio_batch, 1.0 - self.clip_range, 1.0 + self.clip_range)
                policy_loss = - (torch.min(surrogate1, surrogate2)).mean()
            elif self.policy_loss_type == "naive":
                raise NotImplementedError
            elif self.policy_loss_type == "adaptive_kl":
                raise NotImplementedError
            
            #entropy loss
            entropy_loss = - dist_entropy.mean() * self.entropy_coeff
            entropy_loss_value = entropy_loss.item()
            entropy_val = dist_entropy.mean().item()
            policy_loss_value = policy_loss.item()
            tot_policy_loss = policy_loss + entropy_loss

            self.policy_optimizer.zero_grad()
            tot_policy_loss.backward()
            self.policy_optimizer.step()
            update_policy_counts += 1
            
            
        for update_v_step in range(self.train_v_iters):
            #compute value loss
            curr_state_v = self.v_network(obs_batch)
            v_loss = F.mse_loss(curr_state_v, return_batch)
            self.v_optimizer.zero_grad()
            v_loss.backward()
            self.v_optimizer.step()
            v_loss_value = v_loss.detach().cpu().numpy()

        if update_policy_counts > 0:
            return {
                "loss/v": v_loss_value, 
                "loss/policy": policy_loss_value,
                "loss/entropy": entropy_loss_value,
                "misc/entropy": entropy_val,
                "misc/kl_div":approx_kl,
                "misc/policy_updates": update_policy_counts
            }
        else:
            return {
                "loss/v": v_loss_value, 
                "misc/kl_div":approx_kl,
                "misc/policy_updates": update_policy_counts
            }

    @torch.no_grad()
    def estimate_value(self, obs):
        """ Estimate the obs value.
        """
        if len(obs.shape) == 1:
            obs = obs[None,]
    
        if not isinstance(obs, torch.Tensor):
            obs = torch.FloatTensor(obs).to(util.device)
        value = self.v_network(obs)
        return value.detach().cpu().numpy()
    
    @torch.no_grad()
    def select_action(self, obs, deterministic=False):
        if len(obs.shape) == 1:
            ret_single = True
            obs = [obs]
        if type(obs) != torch.tensor:
            obs = torch.FloatTensor(np.array(obs)).to(util.device)
        action, log_prob = itemgetter("action_scaled", "log_prob")(self.policy_network.sample(obs, deterministic=deterministic))
        if ret_single:
            action = action[0]
            log_prob = log_prob[0]
        return {
            'action': action.detach().cpu().numpy(),
            'log_prob' : log_prob
            }




