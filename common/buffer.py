
import numpy as np
import torch
from common.util import device
class REPLAY_BUFFER(object):
    def __init__(self, obs_dim, action_dim, max_buffer_size = 1e5, action_type = gym.spaces.discrete.Discrete):
        self.max_buffer_size = max_buffer_size
        self.curr = 0
        self.obs_buffer = np.zeros((max_buffer_size, obs_dim))
        self.action_buffer = np.zeros((max_buffer_size, ))
        self.next_obs_buffer = np.zeros((max_buffer_size,obs_dim))
        self.reward_buffer = np.zeros((max_buffer_size,))
        self.done_buffer = np.zeros((max_buffer_size,))
        self.max_sample_size = 0
        self.action_type = action_type
    
    def rollout(self, agent, env, max_steps_per_traj, num_trajs):
        for _ in range(num_trajs):
            obs = env.reset()
            for step in range(max_steps_per_traj):
                ac = agent(obs)
                next_obs, reward, done, info = env.step(ac)
                self.add_tuple(obs, ac, next_obs, reward, done)
                if done:
                    break
                pass
                obs = next_obs

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

    def sample_batch(self, batch_size, to_tensor = True):
        #print(self.max_sample_size)
        index = random.sample(range(self.max_sample_size), batch_size)
        obs, action, next_obs, reward, done =  self.obs_buffer[index], \
               self.action_buffer[index],\
               self.next_obs_buffer[index],\
               self.reward_buffer[index],\
               self.done_buffer[index]
        if to_tensor:
            obs = torch.Tensor(obs).to(device)
            action = torch.Tensor(action).to(device)
            if self.action_type == gym.spaces.discrete.Discrete:
                action = action.long()
            next_obs = torch.Tensor(next_obs).to(device)
            reward = torch.Tensor(reward).to(device)
            done = torch.Tensor(done).to(device)
        return obs, action, next_obs, reward, done