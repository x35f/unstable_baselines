import torch
import torch.nn.functional as F
import gym 
from torch import nn
from common.models import BaseAgent
from common.networks import QNetwork, VNetwork, PolicyNetwork, get_optimizer
from common.buffer import ReplayBuffer
import numpy as np
from common.util import device, soft_update_network, hard_update_network

class SACAgent(torch.nn.Module, BaseAgent):
    def __init__(self,observation_space, action_space,**kwargs):
        state_dim = observation_space.shape[0]
        action_dim = action_space.shape[0]
        super(SACAgent, self).__init__()
        #save parameters
        self.args = kwargs
        #initilze networks
        self.q1_network = QNetwork(state_dim, 1,
            hidden_dims = kwargs['q_network']['hidden_dims'],
            act_fn = kwargs['q_network']['act_fn'],
            out_act_fn = kwargs['q_network']['out_act_fn']
            )
        self.q2_network = QNetwork(state_dim, 1,
            hidden_dims = kwargs['q_network']['hidden_dims'],
            act_fn = kwargs['q_network']['act_fn'],
            out_act_fn = kwargs['q_network']['out_act_fn']
            ) 
        self.v_network = VNetwork(state_dim, 1,
            hidden_dims = kwargs['v_network']['hidden_dims'],
            act_fn = kwargs['v_network']['act_fn'],
            out_act_fn = kwargs['v_network']['out_act_fn']
        )
        self.target_v_network = VNetwork(state_dim, 1,
            hidden_dims = kwargs['v_network']['hidden_dims'],
            act_fn = kwargs['v_network']['act_fn'],
            out_act_fn = kwargs['v_network']['out_act_fn']
        )
        self.policy_network = PolicyNetwork(state_dim,action_dim,
            hidden_dims = kwargs['policy_network']['hidden_dims'],
            act_fn = kwargs['policy_network']['act_fn'],
            out_act_fn = kwargs['policy_network']['out_act_fn'],
            deterministic = kwargs['policy_network']['deterministic']
        )

        #sync network parameters
        hard_update_network(self.v_network, self.target_v_network)

        #pass to util.device
        self.q1_network = self.q1_network.to(device)
        self.q2_network = self.q2_network.to(device)
        self.policy_network = self.policy_network.to(device)

        #initialize optimizer
        self.q1_optimizer = get_optimizer(kwargs['q_network']['optimizer_class'], self.q1_network, kwargs['q_network']['learning_rate'])
        self.q2_optimizer = get_optimizer(kwargs['q_network']['optimizer_class'], self.q2_network, kwargs['q_network']['learning_rate'])
        self.v_optimizer = get_optimizer(kwargs['v_network']['optimizer_class'], self.v_network, kwargs['v_network']['learning_rate'])
        self.policy_optimizer = get_optimizer(kwargs['policy_network']['optimizer_class'], self.policy_network, kwargs['policy_network']['learning_rate'])

        #hyper-parameters
        self.gamma = kwargs['gamma']
        self.automatic_entropy_tuning = kwargs['entropy']['automatic_tuning']

        if self.automatic_entropy_tuning is True:
            self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=kwargs['entropy']['learning_rate'])


    def update(self, data_batch):
        state_batch, action_batch, next_state_batch, reward_batch, done_batch = data_batch
        new_curr_state_action, new_curr_state_log_pi, _ = self.policy_network.sample(state_batch)
        new_curr_state_q_value = self.v_network(state_batch, new_curr_state_action)
        next_state_target_v_value = self.target_v_network(next_state_batch)
        curr_state_v_value = self.v_network(state_batch)
        new_curr_state_action_detached = new_curr_state_action.detach()
        curr_state_q1_value = self.q1_network(state_batch, new_curr_state_action)
        curr_state_q2_value = self.q2_network(state_batch, new_curr_state_action)
        
        #compute v loss
        target_v_value = (new_curr_state_q_value - new_curr_state_log_pi).detach()
        v_loss = F.mse_loss(curr_state_v_value, target_v_value)
        v_loss_value = v_loss.detach().cpu().numpy()
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()
        
        #compute q loss
        target_q_value = reward_batch - (1.0 - done_batch) * self.gamma * (next_state_target_v_value.detach())
        q1_loss = F.mse_loss(curr_state_q1_value, target_q_value)
        q2_loss = F.mse_loss(curr_state_q2_value, target_q_value)
        q1_loss_value = q1_loss.detach().cpu().numpy()
        q2_loss_value = q2_loss.detach().cpu().numpy()
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer()
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer()

        #compute policy loss
        min_curr_state_q_value = torch.min(curr_state_q1_value, curr_state_q2_value).detach()
        policy_loss = torch.mean( (self.alpha * new_curr_state_log_pi) - min_curr_state_q_value)
        policy_loss_value = policy_loss.detach().cpu().numpy()
        self.policy_optimizer.zero_grad()
        self.policy_loss.backward()
        self.policy_optimizer.step()

        #compute temperature loss
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (new_curr_state_log_pi + self.target_entropy).detach()).mean()
            alpha_loss_value = alpha_loss.detach.cpu.numpy()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_value = self.alpha.detach().cpu.numpy()
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_value = self.alpha.detach().cpu.numpy()

        return q1_loss_value, q2_loss_value, v_loss_value, policy_loss_value, alpha_loss_value, alpha_value

    def select_action(self, state):
        if type(state) != torch.tensor:
            state = torch.FloatTensor([state]).to(device)
        action_sample = self.policy_network.sample(state)
        print("action_sample", action_sample)
        return action_sample.detach().cpu().numpy()[0]

