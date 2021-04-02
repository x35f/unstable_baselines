import torch
import torch.nn.functional as F
import os
from torch import nn
from common.models import BaseAgent
from common.networks import VNetwork, PolicyNetwork, get_optimizer
import numpy as np
from common import util 

class PPOAgent(torch.nn.Module, BaseAgent):
    def __init__(self,observation_space, action_space,
        gamma,
        beta=1.,
        policy_loss_type="clipped_surrogate",
        use_entropy_loss=False,
        entropy_coeff=0.1,
        c1=1.,
        c2=1.,
        clip_range=0.2, 
        adaptive_kl_coeff=False,

        **kwargs):
        assert policy_loss_type in ['naive', 'clipped_surrogate','adaptive_kl']
        state_dim = observation_space.shape[0]
        action_dim = action_space.shape[0]
        super(PPOAgent, self).__init__()
        #save parameters
        self.args = kwargs
        #initilze networks
        self.v_network = VNetwork(state_dim, 1, **kwargs['v_network'])
        self.policy_network = PolicyNetwork(state_dim, action_space,  ** kwargs['policy_network'])

        #pass to util.device
        self.v_network = self.v_network.to(util.device)
        self.policy_network = self.policy_network.to(util.device)

        #initialize optimizer
        self.v_optimizer = get_optimizer(kwargs['v_network']['optimizer_class'], self.v_network, kwargs['v_network']['learning_rate'])
        self.policy_optimizer = get_optimizer(kwargs['policy_network']['optimizer_class'], self.policy_network, kwargs['policy_network']['learning_rate'])

        #hyper-parameters
        self.gamma = gamma

        #policy loss related hyper-parameters
        self.policy_loss_type = policy_loss_type

        #entropy related
        self.use_entropy_loss = use_entropy_loss
        self.entropy_coeff = entropy_coeff
        self.c1 = c1
        self.c2=c2
        
        #adaptive kl coefficient related parameters
        self.adaptive_kl_coeff = adaptive_kl_coeff
        self.beta = beta

        #clipping related hyper-parameters
        self.clip_range = clip_range

        self.tot_update_count = 0 

    def update(self, data_batch):
        state_batch, action_batch, log_pi_batch, next_state_batch, reward_batch, advantage_batch, return_batch, done_batch = data_batch
        
        curr_state_v = self.v_network(state_batch)
        new_log_pi, dist_entropy = self.policy_network.evaluate_actions(state_batch, action_batch)
        ratio_batch = torch.exp(new_log_pi - log_pi_batch)

        #entropy loss
        entropy_loss = dist_entropy.mean() * self.entropy_coeff
        entropy_loss_value = entropy_loss.item()

        #compute policy loss
        if self.policy_loss_type == "clipped_surrogate":
            surrogate1 = advantage_batch * ratio_batch
            #print(self.clip_range, advantages.shape, ratio_batch.shape)
            surrogate2 =  advantage_batch * torch.clamp(ratio_batch, 1 - self.clip_range, 1 + self.clip_range)
            min_surrogate = - torch.min(surrogate1, surrogate2)
            policy_loss = min_surrogate.mean()
        elif self.policy_loss_type == "naive":
            raise NotImplementedError
        elif self.policy_loss_type == "adaptive_kl":
            raise NotImplementedError
        policy_loss_value = policy_loss.detach().cpu().numpy()
        tot_policy_loss = policy_loss + self.entropy_coeff * entropy_loss
        self.policy_optimizer.zero_grad()
        tot_policy_loss.backward()
        self.policy_optimizer.step()

        #compute value loss
        v_loss = F.mse_loss(curr_state_v, return_batch)
        v_loss_value = v_loss.detach().cpu().numpy()
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()

        
        entropy_val =  torch.mean(dist_entropy).item()
        approx_kl = (log_pi_batch - new_log_pi).mean().item()

        self.tot_update_count += 1
        
        return {
            "loss/v": v_loss_value, 
            "loss/policy": policy_loss_value,
            "loss/entropy": entropy_loss_value,
            "info/entropy": entropy_val,
            "info/kl_div":approx_kl
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


        



