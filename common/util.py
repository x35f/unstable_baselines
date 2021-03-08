import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
import numpy as np
import random
import gym
device = None

def set_device(gpu_id):
    global device
    if gpu_id < 0 or torch.cuda.is_available() == False:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        

def load_config(config_path, **kwargs):
    args_dict = json.load(config_path)
    args_dict = update_parameters(args_dict, kwargs)
    return args_dict

def update_parameters(base_dict, update_dict, print_info = False):
    for k_path in update_dict:
        ks = k_path.split("/")
        #todo: iteration via ks
        if k in base_dict:
            if print_info:
                print("{}:\t\033[32m{}\t->\t{}\033[0m".format(k, base_dict[k],update_dict[k]))
            base_dict[k] = update_dict[k]
        else:
            if print_info:
                print("{}:\t\033[31mnot found\033[0m".format(k))
    return base_dict

def get_value_network_and_optimizer(args):
    network_width = 


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
            global device
            obs = torch.Tensor(obs).to(device)
            action = torch.Tensor(action).to(device)
            if self.action_type == gym.spaces.discrete.Discrete:
                action = action.long()
            next_obs = torch.Tensor(next_obs).to(device)
            reward = torch.Tensor(reward).to(device)
            done = torch.Tensor(done).to(device)
        return obs, action, next_obs, reward, done
        



class LOGGER(object):
    def __init__(self, log_path, prefix="",  warning_level = 3, print_to_terminal = True):
        log_path = self.make_simple_log_path(log_path, prefix)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        self.tb_writer = SummaryWriter(log_path)
        self.log_file_path = os.path.join(log_path,"output.txt")
        self.print_to_terminal = print_to_terminal
        self.warning_level = warning_level
        
    def make_simple_log_path(self, log_path, prefix):
        now = datetime.now()
        suffix = now.strftime("%d_%H:%M")
        pid_str = os.getpid()
        return os.path.join(log_path,"{}-{}-{}".format(prefix, suffix, pid_str))

    def log_str(self, content, level = 4):
        if level < self.warning_level:
            return
        now = datetime.now()
        time_str = now.strftime("%Y-%m-%d %H:%M:%S")
        if self.print_to_terminal:
            print("\033[32m{}\033[0m:\t{}".format(time_str, content))
        with open(self.log_file_path,'a+') as f:
            f.write("{}:\t{}".format(time_str, content))

    def log_var(self, name, val, ite):
        self.tb_writer.add_scalar(name, val, ite)

def soft_update_network(source_network, target_network, tau):
    for target_param, local_param in zip(target_network.parameters(),
                                        source_network.parameters()):
        target_param.data.copy_(tau*local_param.data + (1-tau)*target_param.data)


class SCHEDULER(object):
    def __init__(self, start_value, end_value, duration: int):
        self.start_value = start_value
        self.end_value = end_values
        self.curr = 0
        self.duration = max(1, duration)
    
    def next(self):
        frac = min(self.curr, self.duration) / self.duration
        self.curr = min(self.curr + 1, self.duration)
        return (self.end_value - self.start_value) * frac + self.start_value
            
    def reset(self, idx = 0):
        self.curr = idx
    

