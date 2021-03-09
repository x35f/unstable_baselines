
from abc import abstractmethod
import numpy as np
import torch
from common.util import device
import gym
import random

class BaseBuffer(object):

    def __init__(self):
        pass

    @abstractmethod  
    def rollout(self):
        pass

    @abstractmethod  
    def add_traj(self, obs_list, action_list, next_obs_list, reward_list, done_list):
        pass
    
    @abstractmethod  
    def add_tuple(self):
        pass

    @abstractmethod
    def sample_batch(self):
        pass

class ReplayBuffer(object):
    def __init__(self, obs_space, action_space, max_buffer_size = 1e5, action_type = gym.spaces.discrete.Discrete, **kwargs):
        self.max_buffer_size = max_buffer_size
        self.curr = 0
        obs_dim = obs_space.shape[0]
        action_dim = action_space.shape[0]
        self.obs_buffer = np.zeros((max_buffer_size, obs_dim))
        self.action_buffer = np.zeros((max_buffer_size, action_dim))
        self.next_obs_buffer = np.zeros((max_buffer_size,obs_dim))
        self.reward_buffer = np.zeros((max_buffer_size,))
        self.done_buffer = np.zeros((max_buffer_size,))
        self.max_sample_size = 0
        self.action_type = action_type

    def add_traj(self, obs_list, action_list, next_obs_list, reward_list, done_list):
        for obs, action, next_obs, reward, done in zip(obs_list, action_list, next_obs_list, reward_list, done_list):
            self.add_tuple(obs, action, next_obs, reward, done)
    
    def add_tuple(self, obs, action, next_obs, reward, done):
        self.obs_buffer[self.curr] = obs
        self.action_buffer[self.curr] = action
        self.next_obs_buffer[self.curr] = next_obs
        self.reward_buffer[self.curr] = reward
        self.done_buffer[self.curr] = done
        self.curr = (self.curr+1) % self.max_buffer_size
        self.max_sample_size = min(self.max_sample_size+1, self.max_buffer_size)

    def sample_batch(self, batch_size, to_tensor = True, step_size: int = 1):
        #print(self.max_sample_size):
        if step_size == -1 or step_size > 1: # for td(\lambda) and td(n)
            max_sample_size = np.inf if step_size == -1 else step_size
            index = random.sample(range(self.max_sample_size), batch_size)
            obs_batch = self.obs_buffer[index]
            action_batch, next_obs_batch, reward_batch, done_batch = [[] for _ in range(batch_size)], [[] for _ in range(batch_size)], [[] for _ in range(batch_size)], [[] for _ in range(batch_size)]
            for i, start_index in enumerate(index):
                done = False
                curr_index = start_index
                sampled_num = 0
                while not done and sampled_num < max_sample_size:
                    action_batch[i].append(self.action_buffer[curr_index])
                    next_obs_batch[i].append(self.obs_buffer[curr_index])
                    reward_batch[i].append(self.reward_buffer[curr_index])
                    done_batch[i].append(self.done_buffer[curr_index])
                    curr_index = (curr_index + 1) % self.max_sample_size
                    done = self.done_buffer[curr_index]
        elif step_size == 1:
            batch_size = min(self.max_sample_size, batch_size)
            index = random.sample(range(self.max_sample_size), batch_size)
            obs_batch, action_batch, next_obs_batch, reward_batch, done_batch =  self.obs_buffer[index], \
                self.action_buffer[index],\
                self.next_obs_buffer[index],\
                self.reward_buffer[index],\
                self.done_buffer[index]
        else:
            assert 0, "illegal sample size"
        if to_tensor:
            obs_batch = torch.FloatTensor(obs_batch).to(device)
            action_batch = torch.FloatTensor(action_batch).to(device)
            next_obs_batch = torch.FloatTensor(next_obs_batch).to(device)
            reward_batch = torch.FloatTensor(reward_batch).to(device)
            done_batch = torch.FloatTensor(done_batch).to(device)
        return obs_batch, action_batch, next_obs_batch, reward_batch, done_batch
    