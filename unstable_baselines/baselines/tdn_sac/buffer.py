
from abc import abstractmethod
import numpy as np
import torch
from unstable_baselines.common import util
from unstable_baselines.common.data_structure import *
import gym
import random
from collections import namedtuple

Transition = namedtuple('Transition', ['obs', 'action', 'next_obs', 'reward', 'done'])

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


class TDNReplayBuffer(object):
    def __init__(self, obs_space, action_space, max_buffer_size = 1000000, gamma=0.99,  **kwargs):
        self.max_buffer_size = max_buffer_size
        self.curr = 0
        self.gamma = gamma
        obs_dim = obs_space.shape[0]
        if type(action_space) == gym.spaces.discrete.Discrete:
            action_dim = 1
            #action_dim = action_space.n
            self.discrete_action = True
        elif type(action_space) == gym.spaces.box.Box:
            action_dim = action_space.shape[0]
            self.discrete_action = False
        else:
            assert 0, "unsupported action type"

        self.obs_buffer = np.zeros((max_buffer_size, obs_dim))
        if self.discrete_action:
            self.action_buffer = np.zeros((max_buffer_size, )).astype(np.long)
        else:
            self.action_buffer = np.zeros((max_buffer_size, action_dim))
        self.next_obs_buffer = np.zeros((max_buffer_size,obs_dim))
        self.reward_buffer = np.zeros((max_buffer_size,))
        self.done_buffer = np.zeros((max_buffer_size,))
        self.max_sample_size = 0
        #some redundant info for tdn implementation
        self.max_reward = -np.inf

    def add_traj(self, obs_list, action_list, next_obs_list, reward_list, done_list):
        for obs, action, next_obs, reward, done in zip(obs_list, action_list, next_obs_list, reward_list, done_list):
            self.add_tuple(obs, action, next_obs, reward, done)
    
    def add_tuple(self, obs, action, next_obs, reward, done):
        self.obs_buffer[self.curr] = obs
        self.action_buffer[self.curr] = action
        self.next_obs_buffer[self.curr] = next_obs
        self.reward_buffer[self.curr] = reward
        self.done_buffer[self.curr] = done
        
        #increase pointer
        self.curr = (self.curr + 1) % self.max_buffer_size
        self.max_sample_size = min(self.max_sample_size+1, self.max_buffer_size)

        #some redundant info for tdn implementation
        self.max_reward = max(reward, self.max_reward)


    def sample_batch(self, batch_size, to_tensor = True, step_size = None):
        if step_size is None:
            batch_size = min(self.max_sample_size, batch_size)
            index = random.sample(range(self.max_sample_size), batch_size)
            obs_batch, action_batch, next_obs_batch, reward_batch, done_batch = self.obs_buffer[index], \
                self.action_buffer[index],\
                self.next_obs_buffer[index],\
                self.reward_buffer[index],\
                self.done_buffer[index]
            if to_tensor:
                obs_batch = torch.FloatTensor(obs_batch).to(util.device)
                if self.discrete_action:
                    action_batch = torch.LongTensor(action_batch).to(util.device)
                else:
                    action_batch = torch.FloatTensor(action_batch).to(util.device)
                next_obs_batch = torch.FloatTensor(next_obs_batch).to(util.device)
                reward_batch = torch.FloatTensor(reward_batch).to(util.device).unsqueeze(1)
                done_batch = torch.FloatTensor(done_batch).to(util.device).unsqueeze(1)
            return obs_batch, action_batch, next_obs_batch, reward_batch, done_batch
        else:#td-case
            assert type(step_size) == int and step_size >= 1
            batch_size = min(self.max_sample_size, batch_size)
            if self.curr > step_size:
                valid_indices = list(range(self.curr - step_size)) + list(range(self.curr, self.max_sample_size))
            else:
                valid_indices = list(range(self.curr,self.max_sample_size - (step_size - self.curr)))
            index = random.sample(valid_indices, batch_size)
            obs_batch, action_batch = self.obs_buffer[index], self.action_buffer[index]
            next_obs_batch = np.zeros_like(obs_batch)
            reward_batch = np.zeros((batch_size,))
            done_batch = np.zeros((batch_size, ))
            n_mask_batch = np.zeros((batch_size, )).astype(int)
            for i, start_index in enumerate(index):
                curr_index = start_index
                actual_n = step_size
                for j in range(step_size):
                    buffer_index = (j + start_index) % self.max_sample_size
                    reward_batch[i] += self.reward_buffer[buffer_index] * (self.gamma ** j)
                    if self.done_buffer[buffer_index]:
                        actual_n = j + 1
                        break
                next_index = (start_index + actual_n) % self.max_sample_size
                next_obs_batch[i] = self.next_obs_buffer[(start_index + actual_n - 1) % self.max_sample_size]
                done_batch[i] = self.done_buffer[(start_index + actual_n - 1) % self.max_sample_size]
                n_mask_batch[i] = actual_n
            if to_tensor:
                obs_batch = torch.FloatTensor(obs_batch).to(util.device)
                if self.discrete_action:
                    action_batch = torch.LongTensor(action_batch).to(util.device)
                else:
                    action_batch = torch.FloatTensor(action_batch).to(util.device)
                next_obs_batch = torch.FloatTensor(next_obs_batch).to(util.device)
                reward_batch = torch.FloatTensor(reward_batch).to(util.device).unsqueeze(1)
                done_batch = torch.FloatTensor(done_batch).to(util.device).unsqueeze(1)
                n_mask_batch = torch.FloatTensor(n_mask_batch).to(util.device).unsqueeze(1)
            return obs_batch, action_batch, next_obs_batch, reward_batch, done_batch, n_mask_batch
            

    def sample_specific_buffer(buffer_name: str, batch_size):
        if buffer_name == "obs":
            buffer_to_sample = self.obs_buffer
        elif buffer_name == "action":
            buffer_to_sample = self.action_buffer
        elif buffer_name == "next_obs":
            buffer_to_sample = self.next_obs_buffer
        elif buffer_name == "reward":
            buffer_to_sample = self.reward_buffer
        elif buffer_name == "done":
            buffer_to_sample = self.done_buffer

        batch_size = min(batch_size, self.max_sample_size)
        indices = random.sample(range(self.max_sample_size), batch_size)
    
    def estimate_max_reward(self):
        return self.max_reward

    def estimate_max_value(self, num_samples, value_network):
        num_samples = min(num_samples, self.max_sample_size)
        indices = random.sample(range(self.max_sample_size), num_samples)
        sampled_obs = self.obs_buffer(indices)
        sampled_obs = torch.FloatTensor(sampled_obs).to(util.device)
        value_estimates = value_network(sampled_obs).detach().cpu().numpy()
        max_value = np.max(value_estimates)
        return max_value


    def print_buffer_helper(self, nme, lst, summarize=False, print_curr_ptr = False):
        #for test purpose
        str_to_print = ""
        for i in range(self.max_sample_size):
            if print_curr_ptr:
                str_to_print += "^\t" if self.curr - 1 == i else "\t"  
            elif summarize:
                str_to_print += "{:.02f}\t".format(np.mean(lst[i]))
            else:
                str_to_print += "{:.02f}\t".format(lst[i])
        print("{}:\t{}" .format(nme, str_to_print))

    def print_buffer(self):
        #for test purpose
        self.print_buffer_helper("o",self.obs_buffer, summarize=True)
        #self.print_buffer_helper("a",self.action_buffer, summarize=True)
        self.print_buffer_helper("no",self.next_obs_buffer, summarize=True)
        self.print_buffer_helper("nxt_o",self.n_step_obs_buffer, summarize=True)
        self.print_buffer_helper("r",self.reward_buffer, summarize=True)
        self.print_buffer_helper("dis_r",self.discounted_reward_buffer, summarize=True)
        self.print_buffer_helper("done",self.done_buffer, summarize=True)
        self.print_buffer_helper("nxt_d",self.n_step_done_buffer, summarize=True)
        self.print_buffer_helper("n_count",self.n_count_buffer, print_curr_ptr=True)
        self.print_buffer_helper("index", None, print_curr_ptr=True)
        print("\n")


if __name__ == "__main__":
    from tqdm import tqdm
    
    
    #code for testing discrete action environments
    # env = gym.make("CartPole-v0")
    # obs_space = env.observation_space
    # action_space = env.action_space
    # buffer = ReplayBuffer(obs_space, action_space, max_buffer_size=10)
    # for traj  in tqdm(range(50)):
    #     done = False
    #     obs = env.reset()
    #     while not done:
    #         action = action_space.sample()
    #         next_obs,  reward, done, _ = env.step(action)
    #         buffer.add_tuple(obs, action, next_obs, reward, done)
    #         obs = next_obs

    # code for testing normal buffer
    # env = gym.make("HalfCheetah-v2")
    # obs_space = env.observation_space
    # action_space = env.action_space
    # buffer = ReplayBuffer(obs_space, action_space, max_buffer_size=10)
    # 
    # for traj  in tqdm(range(50)):
    #     done = False
    #     obs = env.reset()
    #     while not done:
    #         action = action_space.sample()
    #         next_obs,  reward, done, _ = env.step(action)
    #         buffer.add_tuple(obs, action, next_obs, reward, done)
    #         obs = next_obs
    # obs_batch, action_batch, next_obs_batch, reward_batch, done_batch = buffer.sample(32,)
    # print(obs_batch[0].shape, action_batch[0].shape, next_obs_batch[0].shape, reward_batch[0].shape, done_batch[0].shape)
    # obs_batch, action_batch, next_obs_batch, reward_batch, done_batch = buffer.sample(32, step_size = 2)
    # print(obs_batch[0].shape, action_batch[0].shape, next_obs_batch[0].shape, reward_batch[0].shape, done_batch[0].shape)
    # obs_batch, action_batch, next_obs_batch, reward_batch, done_batch = buffer.sample(32, step_size = 5)
    # print(obs_batch[0].shape, action_batch[0].shape, next_obs_batch[0].shape, reward_batch[0].shape, done_batch[0].shape)
    # obs_batch, action_batch, next_obs_batch, reward_batch, done_batch = buffer.sample(32, step_size = -1)
    # print(obs_batch[0].shape, action_batch[0].shape, next_obs_batch[0].shape, reward_batch[0].shape, done_batch[0].shape)


    #code for testing td buffer
    env = gym.make("HalfCheetah-v2")
    #env = gym.make("CartPole-v1")
    obs_space = env.observation_space
    action_space = env.action_space
    alpha = 0.8
    max_buffer_size = 16
    max_traj_length = 5
    num_trajs = 4
    ER = PrioritizedReplayBuffer(obs_space, action_space, max_buffer_size, alpha=alpha, epsilon=0.01)

    for traj in tqdm(range(num_trajs)):
        done = False
        obs = env.reset()
        num_steps = 0
        while not done:
            action = action_space.sample()
            next_obs, reward, done, _ = env.step(action)
            num_steps += 1
            if num_steps > max_traj_length:
                done = True
            ER.add_tuple(obs, action, next_obs, reward, done, np.random.random()*10)
            obs = next_obs
            print("step! ----------------------------------------")
            print(ER)

    print("==============================================")
    for i in tqdm(range(4)):
        _, _, _, _, _, p_batch = ER.sample_batch(4, False)
        print(p_batch)

            