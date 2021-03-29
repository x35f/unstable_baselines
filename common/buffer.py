
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
    def __init__(self, obs_space, action_space, max_buffer_size = 1000000, **kwargs):
        self.max_buffer_size = max_buffer_size
        self.curr = 0
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
            self.action_buffer = np.zeros((max_buffer_size, action_dim))
        else:
            self.action_buffer = np.zeros((max_buffer_size, )).astype(np.long)
        self.next_obs_buffer = np.zeros((max_buffer_size,obs_dim))
        self.reward_buffer = np.zeros((max_buffer_size,))
        self.done_buffer = np.zeros((max_buffer_size,))
        self.max_sample_size = 0

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
        self.curr = (self.curr+1) % self.max_buffer_size
        self.max_sample_size = min(self.max_sample_size+1, self.max_buffer_size)

    def sample_batch(self, batch_size, to_tensor = True, step_size: int = 1):
        # batch_size: 
        # to_tensor: if convert to torch.tensor type as pass to util.device
        # step_size: return a list of next states, returns and dones with size n
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
                if self.discrete_action:
                    action_batch = torch.LongTensor(action_batch).to(util.device)
                else:
                    action_batch = torch.FloatTensor(action_batch).to(util.device)
                next_obs_batch = torch.FloatTensor(next_obs_batch).to(util.device)
                reward_batch = torch.FloatTensor(reward_batch).to(util.device).unsqueeze(1)
                done_batch = torch.FloatTensor(done_batch).to(util.device).unsqueeze(1)
            else:
                obs_batch = torch.FloatTensor(obs_batch).to(util.device)
                if self.discrete_action:
                    action_batch = torch.LongTensor(action_batch).to(util.device)
                else:
                    action_batch = torch.FloatTensor(action_batch).to(util.device)
                next_obs_batch = [torch.FloatTensor(next_obs_minibatch).to(util.device) for next_obs_minibatch in next_obs_batch]
                reward_batch = [torch.FloatTensor(reward_minibatch).to(util.device).unsqueeze(1) for reward_minibatch in reward_batch]
                done_batch = [torch.FloatTensor(done_minibatch).to(util.device).unsqueeze(1) for done_minibatch in done_batch]
        return obs_batch, action_batch, next_obs_batch, reward_batch, done_batch

    def print_buffer_helper(self, nme, lst, summarize=False, print_curr_ptr = False):
        #for test purpose
        str_to_print = ""
        for i in range(self.max_sample_size):
            if print_curr_ptr:
                str_to_print += "^\t" if self.curr==i else "\t"  
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
        self.print_buffer_helper("index", None, print_curr_ptr=True)
        print("\n")

class TDReplayBuffer(ReplayBuffer):
    def __init__(self, obs_space, action_space, n, gamma, max_buffer_size = 1000000, **kwargs):
        self.n = n # parameter for td(n)
        self.max_buffer_size = max_buffer_size
        self.gamma = gamma
        obs_dim = obs_space.shape[0]
        if type(action_space) == gym.spaces.discrete.Discrete:
            action_dim = 1
            self.discrete_action = True
        elif type(action_space) == gym.spaces.box.Box:
            action_dim = action_space.shape[0]
            self.discrete_action = False
        else:
            assert 0, "unsupported action type"
        self.obs_buffer = np.zeros((max_buffer_size, obs_dim))
        self.action_buffer = np.zeros((max_buffer_size, action_dim))
        self.next_obs_buffer = np.zeros((max_buffer_size,obs_dim))
        self.reward_buffer = np.zeros((max_buffer_size,))
        self.done_buffer = np.ones((max_buffer_size,))
        self.n_step_obs_buffer = np.zeros((max_buffer_size,obs_dim))
        self.discounted_reward_buffer = np.zeros((max_buffer_size,))
        self.n_step_done_buffer = np.zeros(max_buffer_size,)
        #insert a random state at initialization to avoid bugs when inserting the first state
        self.max_sample_size = 1
        self.curr = 1
    
    def add_tuple(self, obs, action, next_obs, reward, done):
        # store to instant memories
        self.obs_buffer[self.curr] = obs
        self.action_buffer[self.curr] = action
        self.next_obs_buffer[self.curr] = next_obs
        self.reward_buffer[self.curr] = reward
        self.done_buffer[self.curr] = done
        #store precalculated tn(n) info
        self.n_step_obs_buffer[self.curr] = next_obs
        self.discounted_reward_buffer[self.curr] = reward
        self.n_step_done_buffer[self.curr] = 1.
        breaked = False # record if hit the previous trajectory
        for i in range(self.n - 1):
            idx = (self.curr - i - 1) % self.max_sample_size # use max sample size cuz the buffer might not have been full
            if self.done_buffer[idx]: # hit the previous trajecory, break
                breaked = True
                break
            self.discounted_reward_buffer[idx] += (self.gamma ** (i + 1))  * reward
        if not breaked  and not self.done_buffer[(self.curr - self.n) % self.max_sample_size]:# not hit last trajctory, set the n-step-next state for the last state
            self.n_step_obs_buffer[(self.curr - self.n) % self.max_sample_size] = obs
        if done:#set the n-step-next-obs of the previous n states to the current state
            for i in range(self.n - 1):
                idx = (self.curr - i - 1) % self.max_sample_size
                if self.done_buffer[idx]:# hit the last trajectory
                    break
                self.n_step_obs_buffer[idx] = next_obs
                self.n_step_done_buffer[idx] = 1.0
        else:
            prev_idx = (self.curr - 1) % self.max_sample_size
            if not self.done_buffer[prev_idx]:
                self.n_step_done_buffer[prev_idx] = 0.
            for i in range(self.n - 1):
                idx = (self.curr - i - 1) % self.max_sample_size
                if self.done_buffer[idx]:# hit the last trajectory
                    break
                self.n_step_obs_buffer[idx] = next_obs
                self.n_step_done_buffer[idx] = 0.0
            # set the n step ealier done to false
            idx = (self.curr - self.n) % self.max_sample_size

        # another special case is that n > max_sample_size, that might casuse a cyclic visiting of a buffer that has no done states
        # this has been avoided by setting initializing all done states to true
        self.curr = (self.curr+1) % self.max_buffer_size
        self.max_sample_size = min(self.max_sample_size + 1, self.max_buffer_size)

    def sample_batch(self, batch_size, to_tensor=True, n=None):
        if n is not None and n != self.n:
            self.update_td(n)
        # compute valid indices to sample (in case the sampled td tuples are incorrect due to the done states)
        if self.done_buffer[self.curr]:
            # whole trajectories have been recorded
            valid_indices =range(self.max_sample_size)
        elif self.curr > self.n:
            #ignore the previous n tuples
            valid_indices = list(range(self.curr - self.n)) + list(range(self.curr + 1, self.max_sample_size))
        else:
            valid_indices = range(self.curr + 1, self.max_sample_size - (self.n - self.curr))
        batch_size = min(len(valid_indices), batch_size)
        index = random.sample(len(valid_indices), batch_size)
        obs_batch, action_batch, n_step_obs_batch, discounted_reward_batch, n_step_done_batch =\
            self.obs_buffer[index], \
            self.action_buffer[index],\
            self.n_step_obs_buffer[index],\
            self.discounted_reward_buffer[index],\
            self.n_step_done_buffer[index]
        if to_tensor:
            obs_batch = torch.FloatTensor(obs_batch).to(util.device)
            if self.discrete_action:
                action_batch = torch.LongTensor(action_batch).to(util.device)
            else:
                action_batch = torch.FloatTensor(action_batch).to(util.device)
            n_step_obs_batch = torch.FloatTensor(n_step_obs_batch).to(util.device)
            discounted_reward_batch = torch.FloatTensor(discounted_reward_batch).to(util.device).unsqueeze(1)
            n_step_done_batch = torch.FloatTensor(n_step_done_batch).to(util.device).unsqueeze(1)
            
        return obs_batch, action_batch, n_step_obs_batch, discounted_reward_batch, n_step_done_batch

    def update_td(self, n):
        # function: update the current buffer to the new td(n) mode
        print("Updating the current buffer from td \033[32m{} to {}\033[0m".format(self.n, n))
        # reset all discounted_reward_batch, n_step_done_batch and n_step_obs_batch
        self.n_step_obs_buffer = np.zeros_like(self.n_step_obs_buffer)
        self.discounted_reward_buffer = np.zeros_like(self.discounted_reward_buffer)
        self.n_step_done_buffer = np.zeros_like(self.n_step_done_buffer)
        curr = (self.curr - 1) % self.max_sample_size # self.curr points to the index to be overwrite, so decrease by 1
        curr_traj_end_idx = curr # mark the end of the current trajectory
        num_trajs = int(np.sum(self.done_buffer))
        while num_trajs > 0:
            self.n_step_done_buffer[curr_traj_end_idx] = self.done_buffer[curr_traj_end_idx] # set done of the last state
            self.n_step_obs_buffer[curr_traj_end_idx] = self.next_obs_buffer[curr_traj_end_idx] # set the next obs of the last state
            #calculate the length of the current trajectory
            curr_traj_len = 1
            idx = (curr_traj_end_idx - 1) % self.max_sample_size
            while(not self.done_buffer[idx]):
                idx = (idx - 1) % self.max_sample_size
                curr_traj_len += 1
            #backward through the last n states and set the n step done/obs buffer
            for i in range(n - 1):
                idx = (curr_traj_end_idx - i - 1) % self.max_sample_size
                if self.done_buffer[idx]:
                    break
                self.n_step_obs_buffer[idx] = self.next_obs_buffer[curr_traj_end_idx]
                self.n_step_done_buffer[idx] = self.done_buffer[curr_traj_end_idx]
            #accumulate the discounted return
            for i in range(curr_traj_len):
                curr_return = self.reward_buffer[ (curr_traj_end_idx - i) % self.max_sample_size]
                for j in range( min(n, curr_traj_len - i)):
                    target_idx = curr_traj_end_idx - i - j 
                    self.discounted_reward_buffer[target_idx] += (curr_return * (self.gamma ** j))
            # set the n_step_done/obs buffer
            if  curr_traj_len >= n:
                for i in range(curr_traj_len - n): # 3
                    curr_idx = (curr_traj_end_idx - n - i ) % self.max_sample_size
                    if self.done_buffer[curr_idx]:
                        break
                    next_obs_idx = (curr_idx + n) % self.max_sample_size
                    self.n_step_obs_buffer[curr_idx] = self.obs_buffer[next_obs_idx]
            self.print_buffer()
            curr_traj_end_idx = (curr_traj_end_idx - curr_traj_len) % self.max_sample_size
            num_trajs -= 1
        self.n = n
    
    
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
    # obs_batch, action_batch, next_obs_batch, reward_batch, done_batch = buffer.sample_batch(32,)
    # print(obs_batch[0].shape, action_batch[0].shape, next_obs_batch[0].shape, reward_batch[0].shape, done_batch[0].shape)
    # obs_batch, action_batch, next_obs_batch, reward_batch, done_batch = buffer.sample_batch(32, step_size = 2)
    # print(obs_batch[0].shape, action_batch[0].shape, next_obs_batch[0].shape, reward_batch[0].shape, done_batch[0].shape)
    # obs_batch, action_batch, next_obs_batch, reward_batch, done_batch = buffer.sample_batch(32, step_size = 5)
    # print(obs_batch[0].shape, action_batch[0].shape, next_obs_batch[0].shape, reward_batch[0].shape, done_batch[0].shape)
    # obs_batch, action_batch, next_obs_batch, reward_batch, done_batch = buffer.sample_batch(32, step_size = -1)
    # print(obs_batch[0].shape, action_batch[0].shape, next_obs_batch[0].shape, reward_batch[0].shape, done_batch[0].shape)


    #code for testing td buffer
    #env = gym.make("HalfCheetah-v2")
    env = gym.make("CartPole-v1")
    obs_space = env.observation_space
    action_space = env.action_space
    n = 2
    gamma = 0.5
    max_buffer_size = 12
    max_traj_length = 5
    num_trajs = 3
    buffer = TDReplayBuffer(obs_space, action_space, n=n, gamma=gamma, max_buffer_size=max_buffer_size)
    for traj  in tqdm(range(num_trajs)):
        done = False
        obs = env.reset()
        num_steps = 0
        while not done:
            action = action_space.sample()
            next_obs,  reward, done, _ = env.step(action)
            num_steps += 1
            if num_steps > max_traj_length: # break for testing short 
                done = True
            buffer.add_tuple(obs, action, next_obs, reward * 10, done)
            obs = next_obs
            print("step")
            buffer.print_buffer()
        print("inserted traj {}".format(traj))
        buffer.print_buffer()
    print("\033[32m updating td to 3\033[0m")
    buffer.update_td(3)
    buffer.print_buffer()