from unstable_baselines.common import util
from unstable_baselines.common.trainer import BaseTrainer
import numpy as np
from tqdm import tqdm
from time import time
import os
from tqdm import  tqdm

import warnings

class TD3Trainer(BaseTrainer):
    def __init__(self, agent, train_env, eval_env, buffer,
            batch_size,
            policy_delay,
            max_env_steps,
            random_sample_timestep,
            start_update_timestep,
            update_interval,
            load_dir="",
            action_noise_scale = 0.1,
            **kwargs):
        super(TD3Trainer, self).__init__(agent, train_env, eval_env, **kwargs)
        self.agent = agent
        self.buffer = buffer
        #hyperparameters
        self.action_upper_bound  = train_env.action_space.high[0]
        self.action_lower_bound  = train_env.action_space.low[0]
        self.batch_size = batch_size
        self.policy_delay = policy_delay
        self.max_env_steps = max_env_steps
        self.random_sample_timestep = random_sample_timestep
        self.start_update_timestep = start_update_timestep
        self.update_interval = update_interval
        self.action_noise_scale = action_noise_scale
        if load_dir != "" and os.path.exists(load_dir):
            self.agent.load(load_dir)

    def train(self):
        train_traj_returns = [0]
        train_traj_lengths = [0]
        traj_return = 0
        traj_length = 0
        tot_env_steps = 0
        obs = self.train_env.reset()
        done = False
        for env_step in tqdm(range(self.max_env_steps)): # if system is windows, add ascii=True to tqdm parameters to avoid powershell bugs
            self.pre_iter()
            log_infos = {}
            if env_step < self.random_sample_timestep:
                action = self.train_env.action_space.sample()
            else:
                action = self.agent.select_action(obs)['action']
                # add noise and clip action
                action += np.random.randn(action.shape[0]) * self.action_noise_scale
                action = np.clip(action, self.action_lower_bound, self.action_upper_bound)
            next_obs, reward, done, _ = self.train_env.step(action)
            tot_env_steps += 1
            traj_length += 1
            traj_return += reward
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
            
            if env_step < self.start_update_timestep or env_step % self.update_interval != 0:
                continue
            
            for i in range(self.update_interval): # fix sample/update ratio to 1
                update_policy_network = (i + 1) % self.policy_delay == 0 # add + 1 to log policy loss at the last iteration
                data_batch = self.buffer.sample(self.batch_size)
                train_agent_log_infos = self.agent.update(data_batch, update_policy_network=update_policy_network)
            log_infos.update(train_agent_log_infos)
           
            self.post_iter(log_infos, env_step)




            


