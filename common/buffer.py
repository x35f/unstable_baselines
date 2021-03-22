
from abc import abstractmethod
import numpy as np
import torch
from common import util
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
    def __init__(self, obs_space, action_space, max_buffer_size = 1000000, action_type = gym.spaces.discrete.Discrete, **kwargs):
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
        #
        #
        #
        if step_size == -1 or step_size > 1: # for td(\lambda) and td(n)
            max_sample_n = np.inf if step_size == -1 else step_size
            print("samping step size {}, max sample n {}".format(step_size, max_sample_n))
            index = random.sample(range(self.max_sample_size), batch_size)
            obs_batch = self.obs_buffer[index]
            action_batch, next_obs_batch, reward_batch, done_batch = [[] for _ in range(batch_size)], \
                                                            [[] for _ in range(batch_size)], \
                                                            [[] for _ in range(batch_size)], \
                                                            [[] for _ in range(batch_size)]
            sampled_sizes = []
            for i, start_index in enumerate(index):
                done = False
                curr_index = start_index
                sampled_num = 0
                while not done and sampled_num < max_sample_n:
                    action_batch[i].append(self.action_buffer[curr_index])
                    next_obs_batch[i].append(self.obs_buffer[curr_index])
                    reward_batch[i].append(self.reward_buffer[curr_index])
                    done_batch[i].append(self.done_buffer[curr_index])
                    done = self.done_buffer[curr_index]
                    curr_index = (curr_index + 1) % self.max_sample_size
                    sampled_num += 1
                sampled_sizes.append(sampled_num)
        elif step_size == 1:
            batch_size = min(self.max_sample_size, batch_size)
            index = random.sample(range(self.max_sample_size), batch_size)
            obs_batch, action_batch, next_obs_batch, reward_batch, done_batch = self.obs_buffer[index], \
                self.action_buffer[index],\
                self.next_obs_buffer[index],\
                self.reward_buffer[index],\
                self.done_buffer[index]
        else:
            assert 0, "illegal sample size"
        if to_tensor:
            #if sampled normally
            if step_size == 1:
                obs_batch = torch.FloatTensor(obs_batch).to(util.device)
                action_batch = torch.FloatTensor(action_batch).to(util.device)
                next_obs_batch = torch.FloatTensor(next_obs_batch).to(util.device)
                reward_batch = torch.FloatTensor(reward_batch).to(util.device).unsqueeze(1)
                done_batch = torch.FloatTensor(done_batch).to(util.device).unsqueeze(1)
            else:
                obs_batch = torch.FloatTensor(obs_batch).to(util.device)
                action_batch = [torch.FloatTensor(action_minibatch).to(util.device) for action_minibatch in action_batch]
                next_obs_batch = [torch.FloatTensor(next_obs_minibatch).to(util.device) for next_obs_minibatch in next_obs_batch]
                reward_batch = [torch.FloatTensor(reward_minibatch).to(util.device).unsqueeze(1) for reward_minibatch in reward_batch]
                done_batch = [torch.FloatTensor(done_minibatch).to(util.device).unsqueeze(1) for done_minibatch in done_batch]
        return obs_batch, action_batch, next_obs_batch, reward_batch, done_batch



if __name__ == "__main__":
    env = gym.make("HalfCheetah-v2")
    obs_space = env.observation_space
    action_space = env.action_space
    buffer = ReplayBuffer(obs_space, action_space)
    from tqdm import tqdm
    for traj  in tqdm(range(50)):
        done = False
        obs = env.reset()
        while not done:
            action = action_space.sample()
            next_obs,  reward, done, _ = env.step(action)
            buffer.add_tuple(obs, action, next_obs, reward, done)
            obs = next_obs
    obs_batch, action_batch, next_obs_batch, reward_batch, done_batch = buffer.sample_batch(32, step_size = 2)
    print(obs_batch[0].shape, action_batch[0].shape, next_obs_batch[0].shape, reward_batch[0].shape, done_batch[0].shape)
    obs_batch, action_batch, next_obs_batch, reward_batch, done_batch = buffer.sample_batch(32, step_size = 5)
    print(obs_batch[0].shape, action_batch[0].shape, next_obs_batch[0].shape, reward_batch[0].shape, done_batch[0].shape)
    obs_batch, action_batch, next_obs_batch, reward_batch, done_batch = buffer.sample_batch(32, step_size = -1)
    print(obs_batch[0].shape, action_batch[0].shape, next_obs_batch[0].shape, reward_batch[0].shape, done_batch[0].shape)