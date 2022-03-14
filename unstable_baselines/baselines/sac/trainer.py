from unstable_baselines.common.util import second_to_time_str
from unstable_baselines.common.trainer import BaseTrainer
import numpy as np
from tqdm import tqdm
from time import time
import cv2
import os
from tqdm import trange
import torch

class SACTrainer(BaseTrainer):
    def __init__(self, 
            agent, 
            train_env, 
            eval_env, 
            buffer,  
            batch_size,
            max_env_steps,
            start_timestep,
            random_policy_timestep, 
            load_dir="",
            **kwargs):
        super(SACTrainer, self).__init__(agent, train_env, eval_env, **kwargs)
        self.agent = agent
        self.buffer = buffer
        #hyperparameters
        self.batch_size = batch_size
        self.max_env_steps = max_env_steps
        self.start_timestep = start_timestep
        self.random_policy_timestep = random_policy_timestep
        if load_dir != "" and os.path.exists(load_dir):
            self.agent.load(load_dir)

    def train(self):
        train_traj_returns = [0]
        train_traj_lengths = [0]
        tot_env_steps = 0
        traj_return = 0
        traj_length = 0
        done = False
        obs = self.train_env.reset()
        for env_step in trange(self.max_env_steps): # if system is windows, add ascii=True to tqdm parameters to avoid powershell bugs
            self.pre_iter()
            log_infos = {}

            if tot_env_steps < self.random_policy_timestep:
                action = self.train_env.action_space.sample()
            else:
                action = self.agent.select_action(obs)['action']
            next_obs, reward, done, _ = self.train_env.step(action)
            traj_length += 1
            traj_return += reward
            if traj_length >= self.max_trajectory_length:
                done = False
            self.buffer.add_transition(obs, action, next_obs, reward, float(done))
            obs = next_obs
            if done or traj_length >= self.max_trajectory_length:
                obs = self.train_env.reset()
                train_traj_returns.append(traj_return)
                train_traj_lengths.append(traj_length)
                traj_length = 0
                traj_return = 0
            log_infos['performance/train_return'] = train_traj_returns[-1]
            log_infos['performance/train_length'] = train_traj_lengths[-1]
            tot_env_steps += 1
            if tot_env_steps < self.start_timestep:
                continue
    
            data_batch = self.buffer.sample(self.batch_size)
            train_agent_log_infos = self.agent.update(data_batch)
            log_infos.update(train_agent_log_infos)
           
            self.post_iter(log_infos, tot_env_steps)



