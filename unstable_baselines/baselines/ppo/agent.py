from operator import itemgetter
import torch
import torch.nn.functional as F
import os
from torch import nn
from unstable_baselines.common.agents import BaseAgent
from unstable_baselines.common.networks import MLPNetwork, PolicyNetworkFactory, get_optimizer
import numpy as np
from unstable_baselines.common import util 

class PPOAgent(BaseAgent):
    def __init__(self,observation_space, action_space,
            beta: float,
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
        obs_dim = observation_space.shape[0]
        
        #initilze networks
        self.v_network = MLPNetwork(obs_dim, 1, **kwargs['v_network'])
        self.policy_network = PolicyNetworkFactory.get(obs_dim, action_space,  **kwargs['policy_network'])

        #pass to util.device
        self.v_network = self.v_network.to(util.device)
        self.policy_network = self.policy_network.to(util.device)

        #register networks
        self.networks = {
            'v_network': self.v_network,
            'policy_network': self.policy_network
        }

        #initialize optimizer
        self.v_optimizer = get_optimizer(kwargs['v_network']['optimizer_class'], self.v_network, kwargs['v_network']['learning_rate'])
        self.policy_optimizer = get_optimizer(kwargs['policy_network']['optimizer_class'], self.policy_network, kwargs['policy_network']['learning_rate'])

        #hyper-parameters
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

    def update(self, data_batch):
        obs_batch, action_batch, log_prob_batch,  advantage_batch, return_batch = \
            itemgetter("obs", "action", "log_prob", "advantage", "ret")(data_batch)
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
            policy_loss_value = policy_loss.detach().cpu().numpy()
            tot_policy_loss = policy_loss# + entropy_loss

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
    def select_action(self, state, deterministic=False):
        if type(state) != torch.tensor:
            state = torch.FloatTensor(np.array([state])).to(util.device)
        action, log_prob = itemgetter("action_scaled", "log_prob") (self.policy_network.sample(state, deterministic=deterministic))
        return {
            "action": action.detach().cpu().numpy(),
            "log_prob": log_prob.detach().cpu().numpy()
        }



