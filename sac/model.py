import torch
import torch.nn.functional as F
import gym 
import os
from torch import nn
from common.models import BaseAgent
from common.networks import QNetwork, VNetwork, GaussianPolicyNetwork, get_optimizer
from common.buffer import ReplayBuffer
import numpy as np
from common import util 

class SACAgent(torch.nn.Module, BaseAgent):
    def __init__(self,observation_space, action_space,
        update_target_network_interval = 50, 
        target_smoothing_tau = 0.1,
        **kwargs):
        state_dim = observation_space.shape[0]
        action_dim = action_space.shape[0]
        super(SACAgent, self).__init__()
        #save parameters
        self.args = kwargs
        #initilze networks
        self.q1_network = QNetwork(state_dim + action_dim, 1, **kwargs['q_network'])
        self.q2_network = QNetwork(state_dim + action_dim, 1,**kwargs['q_network'])
        self.target_q1_network = QNetwork(state_dim + action_dim, 1,**kwargs['q_network'])
        self.target_q2_network = QNetwork(state_dim + action_dim, 1,**kwargs['q_network'])
        self.policy_network = GaussianPolicyNetwork(state_dim,action_dim, action_space = action_space,  ** kwargs['policy_network'])

        #sync network parameters
        util.hard_update_network(self.q1_network, self.target_q1_network)
        util.hard_update_network(self.q2_network, self.target_q2_network)

        #pass to util.device
        self.q1_network = self.q1_network.to(util.device)
        self.q2_network = self.q2_network.to(util.device)
        self.target_q1_network = self.target_q1_network.to(util.device)
        self.target_q2_network = self.target_q2_network.to(util.device)
        self.policy_network = self.policy_network.to(util.device)

        #initialize optimizer
        self.q1_optimizer = get_optimizer(kwargs['q_network']['optimizer_class'], self.q1_network, kwargs['q_network']['learning_rate'])
        self.q2_optimizer = get_optimizer(kwargs['q_network']['optimizer_class'], self.q2_network, kwargs['q_network']['learning_rate'])
        self.policy_optimizer = get_optimizer(kwargs['policy_network']['optimizer_class'], self.policy_network, kwargs['policy_network']['learning_rate'])

        #hyper-parameters
        self.gamma = kwargs['gamma']
        self.automatic_entropy_tuning = kwargs['entropy']['automatic_tuning']
        self.alpha = 0.2 
        if self.automatic_entropy_tuning is True:
            self.target_entropy = -np.prod(action_space.shape).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=util.device)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=kwargs['entropy']['learning_rate'])
        self.tot_update_count = 0 
        self.update_target_network_interval = update_target_network_interval
        self.target_smoothing_tau = target_smoothing_tau

    def update(self, data_batch):
        state_batch, action_batch, next_state_batch, reward_batch, done_batch = data_batch
        curr_state_q1_value = self.q1_network(state_batch, action_batch)
        curr_state_q2_value = self.q2_network(state_batch, action_batch)
        new_curr_state_action, new_curr_state_log_pi, _ = self.policy_network.sample(state_batch)
        next_state_action, next_state_log_pi, _ = self.policy_network.sample(next_state_batch)

        new_curr_state_q1_value = self.q1_network(state_batch, new_curr_state_action)
        new_curr_state_q2_value = self.q2_network(state_batch, new_curr_state_action)

        next_state_q1_value = self.target_q1_network(next_state_batch, next_state_action)
        next_state_q2_value = self.target_q2_network(next_state_batch, next_state_action)
        next_state_min_q = torch.min(next_state_q1_value, next_state_q2_value)
        target_q = (next_state_min_q - self.alpha * next_state_log_pi)
        target_q = reward_batch + self.gamma * (1. - done_batch) * target_q

        new_min_curr_state_q_value = torch.min(new_curr_state_q1_value, new_curr_state_q2_value)

        #compute q loss
        q1_loss = F.mse_loss(curr_state_q1_value, target_q.detach())
        q2_loss = F.mse_loss(curr_state_q2_value, target_q.detach())
        q1_loss_value = q1_loss.detach().cpu().numpy()
        q2_loss_value = q2_loss.detach().cpu().numpy()
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        #compute policy loss
        policy_loss = ((self.alpha * new_curr_state_log_pi) - new_min_curr_state_q_value).mean()
        policy_loss_value = policy_loss.detach().cpu().numpy()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        #compute entropy loss
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (new_curr_state_log_pi + self.target_entropy).detach()).mean()
            alpha_loss_value = alpha_loss.detach().cpu().numpy()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_value = self.alpha.detach().cpu().numpy()
        else:
            alpha_loss_value = 0.
            alpha_value = self.alpha
        self.tot_update_count += 1
        return q1_loss_value, q2_loss_value, policy_loss_value, alpha_loss_value, alpha_value

    def try_update_target_network(self):
        if self.tot_update_count % self.update_target_network_interval == 0:
            util.soft_update_network(self.q1_network, self.target_q1_network, self.target_smoothing_tau)
            util.soft_update_network(self.q2_network, self.target_q2_network, self.target_smoothing_tau)
            
    def select_action(self, state, evaluate=False):
        if type(state) != torch.tensor:
            state = torch.FloatTensor([state]).to(util.device)
        action, log_prob, mean = self.policy_network.sample(state)
        if evaluate:
            return mean.detach().cpu().numpy()[0]
        else:
            return action.detach().cpu().numpy()[0]


    def save_model(self, target_dir, ite):
        assert os.path.isdir(target_dir) 
        target_dir = os.mkdir(os.path.join(target_dir, "ite_{}".format(ite)))
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        #save q networks 
        save_path = os.path.join(target_dir, "Q_network_1.pt")
        torch.save(self.q1_network, save_path)
        save_path = os.path.join(target_dir, "Q_network_2.pt")
        torch.save(self.q2_network, save_path)
        #save policy network
        save_path = os.path.join(target_dir, "policy_network.pt")
        torch.save(self.policy_network, save_path)


    def load_model(self, model_dir):
        q1_network_path = os.path.join(model_dir, "Q_network_1.pt")
        self.q1_network.load_state_dict(torch.load(q1_network_path))
        q2_network_path = os.path.join(model_dir, "Q_network_2.pt")
        self.q2_network.load_state_dict(torch.load(q2_network_path))
        policy_network_path = os.path.join(model_dir, "policy_network.pt")
        self.policy_network.load_state_dict(torch.load(policy_network_path))


        



