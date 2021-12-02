import torch
import torch.nn.functional as F
import gym 
import os
from torch import nn
from unstable_baselines.common.agents import BaseAgent
from unstable_baselines.common.networks import MLPNetwork, PolicyNetwork, get_optimizer
from unstable_baselines.common.buffer import ReplayBuffer
import numpy as np
from unstable_baselines.common import util 
from unstable_baselines.common.maths import product_of_gaussians

class PEARLAgent(torch.nn.Module, BaseAgent):
    def __init__(self,observation_space, action_space,
        update_target_network_interval=50, 
        target_smoothing_tau=0.1,
        alpha=0.2,
        kl_lambda=0.05,
        policy_mean_reg_weight=1e-3,
        policy_std_reg_weight=1e-3,
        policy_pre_activation_weight=0.0,
        **kwargs):
        obs_dim = observation_space.shape[0]
        action_dim = action_space.shape[0]
        super(PEARLAgent, self).__init__()
        #save parameters
        self.args = kwargs

        #initilze networks
        self.latent_dim = kwargs['latent_dim']
        self.q1_network = MLPNetwork(obs_dim + action_dim + self.latent_dim, 1, **kwargs['q_network'])
        self.q2_network = MLPNetwork(obs_dim + action_dim + self.latent_dim, 1,**kwargs['q_network'])
        self.v_network = MLPNetwork(obs_dim + self.latent_dim, 1,**kwargs['v_network'])
        self.target_v_network = MLPNetwork(obs_dim + self.latent_dim, 1,**kwargs['v_network'])
        self.policy_network = PolicyNetwork(obs_dim + self.latent_dim, action_space,  ** kwargs['policy_network'])
        self.use_next_obs_in_context = kwargs['use_next_obs_in_context']
        if self.use_next_obs_in_context:
            context_encoder_input_dim = 2 * obs_dim + action_dim + 1
        else:
            context_encoder_input_dim =  obs_dim + action_dim + 1
        self.use_information_bottleneck = kwargs['use_information_bottleneck']
        if self.use_information_bottleneck:
            context_encoder_output_dim = kwargs['latent_dim'] * 2
        else:
            context_encoder_output_dim = kwargs['latent_dim']
        self.context_encoder_network = MLPNetwork(context_encoder_input_dim, context_encoder_output_dim, **kwargs['context_encoder_network'])

        #pass to util.device
        self.q1_network = self.q1_network.to(util.device)
        self.q2_network = self.q2_network.to(util.device)
        self.v_network = self.v_network.to(util.device)
        self.target_v_network = self.target_v_network.to(util.device)
        self.policy_network = self.policy_network.to(util.device)
        self.context_encoder_network = self.context_encoder_network.to(util.device)

        #register networks
        self.networks = {
            'q1_network': self.q1_network,
            'q2_network': self.q2_network,
            'v_network': self.v_network,
            'target_v_network': self.target_v_network,
            'policy_network': self.policy_network,
            'context_encoder_network': self.context_encoder_network
        }

        #initialize optimizer
        self.q1_optimizer = get_optimizer(kwargs['q_network']['optimizer_class'], self.q1_network, kwargs['q_network']['learning_rate'])
        self.q2_optimizer = get_optimizer(kwargs['q_network']['optimizer_class'], self.q2_network, kwargs['q_network']['learning_rate'])
        self.v_optimizer = get_optimizer(kwargs['v_network']['optimizer_class'], self.v_network, kwargs['v_network']['learning_rate'])
        self.policy_optimizer = get_optimizer(kwargs['policy_network']['optimizer_class'], self.policy_network, kwargs['policy_network']['learning_rate'])
        self.context_encoder_optimizer = get_optimizer(kwargs['context_encoder_network']['optimizer_class'], self.policy_network, kwargs['context_encoder_network']['learning_rate'])

        #hyper-parameters
        self.gamma = kwargs['gamma']
        self.alpha = alpha
        self.kl_lambda = kl_lambda
        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.policy_std_reg_weight = policy_std_reg_weight
        self.policy_pre_activation_weight = policy_pre_activation_weight
        self.tot_update_count = 0 
        self.update_target_network_interval = update_target_network_interval
        self.target_smoothing_tau = target_smoothing_tau

    def update(self, context_batch, data_batch):
        num_tasks = len(context_batch)

        obs_batch, action_batch, next_obs_batch, reward_batch, done_batch = data_batch
        
        # infer z from context
        z_means, z_vars = self.infer_z_posterior(context_batch)
        task_z_batch = self.sample_z_from_posterior(z_means, z_vars)

        # expand z to concatenate with obs, action batches
        num_tasks, batch_size, obs_dim = obs_batch.shape

        #flatten obs
        obs_batch = obs_batch.view(num_tasks * batch_size, -1)
        action_batch = action_batch.view(num_tasks * batch_size, -1)
        next_obs_batch = next_obs_batch.view(num_tasks * batch_size, -1)
        reward_batch = reward_batch.view(num_tasks * batch_size, -1)
        done_batch = done_batch.view(num_tasks * batch_size, -1)
        
        #expand z to match obs batch
        task_z_batch = [task_z.repeat(batch_size, 1 ) for task_z in task_z_batch]
        task_z_batch = torch.cat(task_z_batch, dim=0)

        # get new policy output
        policy_input = torch.cat([obs_batch, task_z_batch.detach()], dim=1)
        new_action_samples, new_action_log_probs, new_action_means, extra_infos = self.policy_network.sample(policy_input)
        new_action_log_stds = extra_infos['action_std']
        pre_tanh_value = extra_infos['pre_tanh_value']

        
        curr_state_q1_value = self.q1_network(torch.cat([obs_batch, action_batch, task_z_batch], dim=1))
        curr_state_q2_value = self.q2_network(torch.cat([obs_batch, action_batch, task_z_batch], dim=1))
        curr_state_v_value = self.v_network(torch.cat([obs_batch, task_z_batch.detach()], dim=1))
        with torch.no_grad():
            target_v_value = self.target_v_network(torch.cat([next_obs_batch, task_z_batch.detach()], dim=1))
        target_q_value = reward_batch + (1 - done_batch) * self.gamma * target_v_value

        #compute losses

        #compute kl loss if use information bottleneck for context optimizer
        prior = torch.distributions.Normal(torch.zeros(self.latent_dim).to(util.device), torch.ones(self.latent_dim).to(util.device))
        posteriors = [torch.distributions.Normal(mu, torch.sqrt(var)) for mu, var in zip(torch.unbind(z_means), torch.unbind(torch.exp(z_vars)))] #todo: inpect std
        kl_divs = [torch.distributions.kl.kl_divergence(post, prior) for post in posteriors]
        kl_div = torch.sum(torch.stack(kl_divs))
        kl_loss = self.kl_lambda * kl_div

        #compute q loss
        q1_loss = torch.mean((curr_state_q1_value - target_q_value) ** 2) 
        q2_loss =  torch.mean((curr_state_q2_value - target_q_value) ** 2)
        q_loss = q1_loss + q2_loss
        
        #conpute encoder loss
        if self.use_information_bottleneck:
            kl_loss.backward(retain_graph=True)

        # compute loss w.r.t value function
        new_action_samples_detached = new_action_samples.detach()
        new_curr_state_q1_value = self.q1_network(torch.cat([obs_batch, new_action_samples_detached, task_z_batch.detach()], dim=1))
        new_curr_state_q2_value = self.q2_network(torch.cat([obs_batch, new_action_samples_detached, task_z_batch.detach()], dim=1))
        new_min_q = torch.min(new_curr_state_q1_value, new_curr_state_q2_value)
        new_target_v_value = (new_min_q - new_action_log_probs).detach()
        v_loss = torch.mean((new_target_v_value - curr_state_v_value) **2)

        #compute loss w.r.t policy function
        target_prob_loss = torch.mean(new_action_log_probs - new_min_q)
        
        mean_reg_loss = torch.mean(self.policy_mean_reg_weight * (new_action_means ** 2))
        std_reg_loss = torch.mean(self.policy_std_reg_weight * (new_action_log_stds ** 2))
        pre_activation_reg_loss = self.policy_pre_activation_weight * ((pre_tanh_value ** 2).sum(dim=1).mean())
        policy_loss = target_prob_loss + mean_reg_loss + std_reg_loss + pre_activation_reg_loss
        #policy_loss = pre_activation_reg_loss

        #backward losses, then update networks
        self.context_encoder_optimizer.zero_grad()
        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        self.policy_optimizer.zero_grad()
        self.v_optimizer.zero_grad()

        policy_loss.backward()
        q_loss.backward()
        v_loss.backward()
        
        self.policy_optimizer.step()
        self.context_encoder_optimizer.step()
        self.q1_optimizer.step()
        self.q2_optimizer.step()
        self.v_optimizer.step()

        #update target v network
        self.try_update_target_network()
        
        kl_subject = "loss/kl_div" if self.use_information_bottleneck else "stats/kl_div"
        return {
            "loss/q1": q1_loss.item(), 
            "loss/q2": q2_loss.item(), 
            "loss/v": v_loss.item(),
            "loss/policy": policy_loss.item(), 
            kl_subject: kl_loss.item(), 
            "stats/train_z_mean": -1, #todo
            "stats/train_z_std": -1 #todo
        }
        


    def infer_z_posterior(self, context_batch):
        num_tasks, batch_size, context_dim = context_batch.shape
        context_batch = context_batch.view(num_tasks * batch_size, context_dim)
        z_params = self.context_encoder_network(context_batch)
        z_params = z_params.view(num_tasks, batch_size, -1)
        if self.use_information_bottleneck:
            z_mean = z_params[..., :self.latent_dim]
            z_sigma_squared = nn.functional.softplus(z_params[..., self.latent_dim:])
            z_params = [product_of_gaussians(mean, std) for mean, std in zip(torch.unbind(z_mean), torch.unbind(z_sigma_squared))]
            z_means = torch.stack([p[0] for p in z_params])
            z_vars = torch.stack([p[1] for p in z_params])
        else:
            z_means = torch.mean(z_params, dim=1)
            z_vars = torch.zeros_like(z_means)
        return z_means, z_vars
        

    def sample_z_from_posterior(self, z_means, z_vars):
        if self.use_information_bottleneck:
            posteriors = [torch.distributions.Normal(m, torch.sqrt(s)) for m, s in zip(torch.unbind(z_means), torch.unbind(z_vars))]
            z = [d.rsample() for d in posteriors]
            z = torch.stack(z)
        else:
            z = self.z_means
        z = z.to(util.device)
        return z

    def set_z(self, z):
        self.z = z    
        
    def clear_z(self, num_tasks=1):
        self.z = torch.zeros((num_tasks, self.latent_dim)).to(util.device)

    def try_update_target_network(self):
        if self.tot_update_count % self.update_target_network_interval == 0:
            util.soft_update_network(self.v_network, self.target_v_network, self.target_smoothing_tau)
            
    def select_action(self, state, z, deterministic=False):
        if type(state) != torch.tensor:
            state = torch.FloatTensor(np.array([state])).to(util.device)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        policy_network_input = torch.cat([state, z.detach()], dim=1)
        action, log_prob, mean, std = self.policy_network.sample(policy_network_input)
        if deterministic:
            return mean.detach().cpu().numpy()[0], log_prob
        else:
            return action.detach().cpu().numpy()[0], log_prob
