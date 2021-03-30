from common.buffer import ReplayBuffer, TDReplayBuffer
import numpy as np
from common import util
import torch
import random
from common import util

def rollout(env, agent, max_env_steps, gamma=0.99, max_trajectories=-1, max_traj_length=1000, n=1):
    #max_env_steps: max environments to sample
    #max_trajectories: max trajectories to sample
    #max_traj_length: max length of each trajectory
    #n: for td learning
    traj_rewards = []
    traj_lengths = []
    
    max_rollout_buffer_size = max_env_steps + max_traj_length # in case an additional full trajectory is sampled
    # if n == 1:
    #     #naive buffer case
    #     rollout_buffer = NaiveRollout(env.observation_space, env.action_space, gamma=gamma, max_buffer_size=max_rollout_buffer_size)
    # else:
    #     #td buffer case
    rollout_buffer = TDRollout(env.observation_space, env.action_space, n=n, gamma=gamma, max_buffer_size=max_rollout_buffer_size)
    if max_trajectories == -1:
        max_trajectories = np.inf
    tot_env_steps = 0
    tot_trajectories = 0
    while(tot_trajectories < max_trajectories):
        states, actions, log_pis, next_states, rewards, dones = rollout_trajectory(env, agent, max_traj_length)
        rollout_buffer.add_traj(states, actions,log_pis, next_states, rewards, dones)
        tot_env_steps += len(states)
        traj_rewards.append(np.sum(rewards))
        traj_lengths.append(len(states))

        if tot_env_steps > max_env_steps:
            break
        tot_trajectories += 1
    
    rollout_buffer.finalize() #convert to tensor type, calculate all return to go

    return rollout_buffer, np.mean(traj_rewards), np.mean(traj_lengths)


def rollout_trajectory(env, agent, max_traj_length):
    states, actions, log_pis, next_states, rewards, dones = [], [], [], [], [], []
    state = env.reset()
    done = False
    traj_length = 0
    while not done:
        action, log_pi = agent.act(state)
        next_state, reward, done, info = env.step(action)
        traj_length += 1
        if traj_length >= max_traj_length:
            done = 1.
        states.append(state)
        actions.append(action)
        log_pis.append(log_pi)
        next_states.append(next_state)
        rewards.append(reward)
        dones.append(done)
        state = next_state
        if done:
            break
    return states, actions, log_pis, next_states, rewards, dones



class NaiveRollout(object):
    def __init__(self, obs_space, action_space, gamma = 0.99, max_buffer_size = 1000000, **kwargs):
        self.max_buffer_size = max_buffer_size
        self.curr = 0
        obs_dim = obs_space.shape[0]
        action_dim = action_space.shape[0]
        self.obs_buffer = np.zeros((max_buffer_size, obs_dim))
        self.action_buffer = np.zeros((max_buffer_size, action_dim))
        self.log_pi_buffer = np.zeros((max_buffer_size,))
        self.next_obs_buffer = np.zeros((max_buffer_size,obs_dim))
        self.reward_buffer = np.zeros((max_buffer_size,))
        self.done_buffer = np.zeros((max_buffer_size,))
        self.max_sample_size = 0
        self.gamma = gamma

    def add_traj(self, obs_list, action_list, log_pi_list, next_obs_list, reward_list, done_list):
        for obs, action, log_pi, next_obs, reward, done in zip(obs_list, action_list, log_pi_list, next_obs_list, reward_list, done_list):
            self.add_tuple(obs, action, log_pi, next_obs, reward, done)
    
    @property
    def size(self):
        return self.max_sample_size

    def add_tuple(self, obs, action, log_pi, next_obs, reward, done):
        self.obs_buffer[self.curr] = obs
        self.action_buffer[self.curr] = action
        self.log_pi_buffer[self.curr] = log_pi
        self.next_obs_buffer[self.curr] = next_obs
        self.reward_buffer[self.curr] = reward
        self.done_buffer[self.curr] = done
        self.curr = (self.curr+1) % self.max_buffer_size
        self.max_sample_size = min(self.max_sample_size+1, self.max_buffer_size)

    def finalize(self):
        #calculate future return for all trajectories
        self.future_return_buffer = np.zeros(self.max_sample_size)
        curr_return = 0.
        for i in range(self.max_sample_size):
            idx = self.max_sample_size - 1 - i
            if self.done_buffer[idx]:
                curr_return = 0.
            curr_return = self.reward_buffer[idx] + self.gamma * curr_return
            self.future_return_buffer[idx] = curr_return

        #convert to tensor and pass data to device
        self.obs_buffer = torch.FloatTensor(self.obs_buffer[:self.max_sample_size]).to(util.device)
        self.action_buffer = torch.FloatTensor(self.action_buffer[:self.max_sample_size]).to(util.device)
        self.log_pi_buffer = torch.FloatTensor(self.log_pi_buffer[:self.max_sample_size]).to(util.device)
        self.next_obs_buffer = torch.FloatTensor(self.next_obs_buffer[:self.max_sample_size]).to(util.device)
        self.reward_buffer = torch.FloatTensor(self.reward_buffer[:self.max_sample_size]).to(util.device).unsqueeze(1)
        self.future_return_buffer = torch.FloatTensor(self.future_return_buffer[:self.max_sample_size]).to(util.device).unsqueeze(1)
        self.done_buffer = torch.FloatTensor(self.done_buffer[:self.max_sample_size]).to(util.device).unsqueeze(1)


    def sample_batch(self, batch_size, to_tensor = True, step_size: int = 1):
        # batch_size: 
        # to_tensor: if convert to torch.tensor type as pass to util.device
        # step_size: return a list of next states, returns and dones with size n
        batch_size = min(self.max_sample_size, batch_size)
        index = random.sample(range(self.max_sample_size), batch_size)
        obs_batch, action_batch, log_pi_batch, next_obs_batch, reward_batch, future_return_batch, done_batch = \
            self.obs_buffer[index], \
            self.action_buffer[index],\
            self.log_pi_buffer[index],\
            self.next_obs_buffer[index],\
            self.reward_buffer[index],\
            self.future_return_buffer[index],\
            self.done_buffer[index]
        return obs_batch, action_batch, log_pi_batch, next_obs_batch, reward_batch, future_return_batch, done_batch

class TDRollout(object):
    def __init__(self, obs_space, action_space, n, gamma, max_buffer_size = 1000000, **kwargs):
        self.n = n # parameter for td(n)
        self.max_buffer_size = max_buffer_size
        self.gamma = gamma
        obs_dim = obs_space.shape[0]
        action_dim = action_space.shape[0]
        self.obs_buffer = np.zeros((max_buffer_size, obs_dim))
        self.action_buffer = np.zeros((max_buffer_size, action_dim))
        self.log_pi_buffer = np.zeros((max_buffer_size,))
        self.reward_buffer = np.zeros((max_buffer_size,))
        self.done_buffer = np.ones((max_buffer_size,))
        self.n_step_obs_buffer = np.zeros((max_buffer_size,obs_dim))
        self.discounted_reward_buffer = np.zeros((max_buffer_size,))
        self.n_step_done_buffer = np.zeros(max_buffer_size,)
        #insert a random state at initialization to avoid bugs when inserting the first state
        self.max_sample_size = 1
        self.curr = 1
    
    @property
    def size(self):
        return self.max_sample_size

    def add_traj(self, obs_list, action_list, log_pi_list, next_obs_list, reward_list, done_list):
        for obs, action, log_pi, next_obs, reward, done in zip(obs_list, action_list, log_pi_list, next_obs_list, reward_list, done_list):
            self.add_tuple(obs, action, log_pi, next_obs, reward, done)
    
    def add_tuple(self, obs, action, log_pi, next_obs, reward, done):
        # store to instant memories
        self.obs_buffer[self.curr] = obs
        self.action_buffer[self.curr] = action
        self.log_pi_buffer[self.curr] = log_pi
        #self.next_obs_buffer[self.curr] = next_obs
        self.reward_buffer[self.curr] = reward
        self.done_buffer[self.curr] = done
        #store precalculated tn(n) info
        self.n_step_obs_buffer[self.curr] = next_obs
        self.discounted_reward_buffer[self.curr] = reward
        self.n_step_done_buffer[self.curr] = 0.
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
            self.n_step_done_buffer[self.curr] = 1.0 
            for i in range(self.n - 1):
                idx = (self.curr - i -1) % self.max_sample_size
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



    def finalize(self):
        #convert to tensor and pass data to device
        self.obs_buffer = torch.FloatTensor(self.obs_buffer[:self.max_sample_size]).to(util.device)
        self.action_buffer = torch.FloatTensor(self.action_buffer[:self.max_sample_size]).to(util.device)
        self.log_pi_buffer = torch.FloatTensor(self.log_pi_buffer[:self.max_sample_size]).to(util.device)
        self.n_step_obs_buffer = torch.FloatTensor(self.n_step_obs_buffer[:self.max_sample_size]).to(util.device)
        self.reward_buffer = torch.FloatTensor(self.reward_buffer[:self.max_sample_size]).to(util.device)
        self.discounted_reward_buffer = torch.FloatTensor(self.discounted_reward_buffer[:self.max_sample_size]).to(util.device).unsqueeze(1)
        self.n_step_done_buffer = torch.FloatTensor(self.n_step_done_buffer[:self.max_sample_size]).to(util.device).unsqueeze(1)

    def sample_batch(self, batch_size, to_tensor = True, step_size: int = 1):
        # batch_size: 
        # to_tensor: if convert to torch.tensor type as pass to util.device
        # step_size: return a list of next states, returns and dones with size n
        batch_size = min(self.max_sample_size, batch_size)
        index = random.sample(range(self.max_sample_size), batch_size)
        obs_batch, action_batch, log_pi_batch, next_obs_batch, reward_batch, discounted_reward_batch, done_batch = \
            self.obs_buffer[index], \
            self.action_buffer[index],\
            self.log_pi_buffer[index],\
            self.n_step_obs_buffer[index],\
            self.reward_buffer[index],\
            self.discounted_reward_buffer[index],\
            self.done_buffer[index]
        return obs_batch, action_batch, log_pi_batch, next_obs_batch, reward_batch, discounted_reward_batch, done_batch
