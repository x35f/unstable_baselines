import torch
from util import REPLAY_BUFFER, soft_update_network
import gym 
from torch import nn
from common import util
from common.networks import ValueNetwork, PolicyNetwork, get_optimizer
import random
import numpy as np


class BaseAgent(torch.nn.Module):
    def __init__(self,**kwargs):
        super(BaseAgent,self).__init__(kwargs)
class SACAgent(torch.nn.Module):
    def __init__(self, logger,env,args):
        super(SACAgent, self).__init__()
        assert(type(env.action_space) == gym.spaces.discrete.Discrete)
        #save parameters
        self.logger = logger
        self.env = env
        self.args = args
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        #initilize replay buffer
        self.replay_buffer = REPLAY_BUFFER(obs_dim, action_dim, args.max_buffer_size)
        #initilze networks
        self.q1_network = ValueNetwork(obs_dim, action_dim,
            hidden_dims = args['q_network']['hidden_dims'],
            act_fn = args['q_network']['act_fn'],
            out_act_fn = args['q_network']['out_act_fn']
            )
        self.q2_network = ValueNetwork(obs_dim, action_dim,
            hidden_dims = args['q_network']['hidden_dims'],
            act_fn = args['q_network']['act_fn'],
            out_act_fn = args['q_network']['out_act_fn']
            ) 
        self.policy_network = PolicyNetwork(obs_dim,action_dim,
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


    def train(self):
        tot_num_update = 0
        episode_reward = 0
        episode_length = 0
        done = False
        obs = env.reset()
        for ite in range(self.max_iteration):
            action = 

            if epoch % self.test_interval == 0:
                self.test(tot_num_update)
                #self.logger.log_str("Epoch:\t{}\tLoss:\t{}\tReturn:\t{}".format(epoch,loss_val,mean_reward))

    def test(self,ite):
        traj_rewards = []
        for traj_id in range(self.num_test_trajs):
            obs = self.env.reset()
            traj_reward = 0
            #rollout in environment
            for step in range(self.max_traj_length):
                action = self.select_action(obs)
                next_obs, reward, done, info = self.env.step(action)
                traj_reward += reward
                obs = next_obs
                if done:
                    break
            traj_rewards.append(traj_reward)
        mean_reward = np.mean(traj_rewards)
        #print(traj_rewards)
        self.logger.log_var("performance/test_return",mean_reward, ite )
        self.logger.log_str("Iteration {}\t:\tTest average return {:02f}\t{}".format(ite, mean_reward,traj_rewards))
            
    def select_action(self, obs):
        ob = torch.tensor(obs).to(util.device).unsqueeze(0).float()
        Q_values = self.Q_network(ob)
        Q_values, action_indices = torch.max(Q_values, dim=1)
        return action_indices.detach().cpu().numpy()[0]
