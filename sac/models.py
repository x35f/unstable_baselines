import torch
import gym 
from torch import nn
from common.models import BaseAgent
from common.networks import ValueNetwork, PolicyNetwork, get_optimizer
import numpy as np

class SACAgent(torch.nn.Module, BaseAgent):
    def __init__(self,state_dim, action_dim,args):
        super(SACAgent, self).__init__()
        assert type(env.action_space) == gym.spaces.discrete.Discrete 
        #save parameters
        self.args = args
        #initilize replay buffer
        self.replay_buffer = REPLAY_BUFFER(state_dim, action_dim, args.max_buffer_size)
        #initilze networks
        self.q1_network = ValueNetwork(state_dim, action_dim,
            hidden_dims = args['q_network']['hidden_dims'],
            act_fn = args['q_network']['act_fn'],
            out_act_fn = args['q_network']['out_act_fn']
            )
        self.q2_network = ValueNetwork(state_dim, action_dim,
            hidden_dims = args['q_network']['hidden_dims'],
            act_fn = args['q_network']['act_fn'],
            out_act_fn = args['q_network']['out_act_fn']
            ) 
        self.policy_network = PolicyNetwork(state_dim,action_dim,
            hidden_dims = args['policy_network']['hidden_dims'],
            act_fn = args['policy_network']['act_fn'],
            out_act_fn = args['policy_network']['out_act_fn'],
            deterministic = args['policy_network']['deterministic']
        )
        #pass to util.device
        self.q1_network = self.q1_network.to(util.device)
        self.q2_network = self.q2_network.to(util.device)
        self.policy_network = self.policy_network.to(util.device)

        #initialize optimizer
        self.q1_optimizer = get_optimizer(args['q_network']['optimizer_class'], self.q1_network, args['q_network']['learning_rate'])
        self.q2_optimizer = get_optimizer(args['q_network']['optimizer_class'], self.q2_network, args['q_network']['learning_rate'])
        self.policy_optimier = get_optimizer(args['policy_network']['optimizer_class'], self.policy_network, args['policy_network']['learning_rate'])

        #hyper parameters
        self.max_iteration = args['max_iteration']
        self.update_per_iteration = args['update_per_iteration']
        self.batch_size = args['batch_size']
        self.max_traj_length = args['max_trajectory_length']
        self.test_interval = args['test_interval']
        self.num_test_trajs = args['num_test_trajectories']


    def update(self, data_batch):
        state_batch, action_batch, next_obs_batch, reward_batch, done_batch = data_batch

    def select_action(self, state):
        return self.policy_network.sample(state)

