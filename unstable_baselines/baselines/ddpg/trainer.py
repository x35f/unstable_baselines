from unstable_baselines.common.trainer import BaseTrainer
import numpy as np
import os
from tqdm import  trange

import warnings

class DDPGTrainer(BaseTrainer):
    def __init__(self, agent, train_env, eval_env, buffer, load_dir,
            batch_size,
            max_env_steps,
            random_sample_timestep,
            start_update_timestep,
            update_interval,
            action_noise_scale,
            **kwargs):
        super(DDPGTrainer, self).__init__(agent, train_env, eval_env, **kwargs)
        self.buffer = buffer
        #hyperparameters
        self.action_upper_bound  = train_env.action_space.high[0]
        self.action_lower_bound  = train_env.action_space.low[0]
        self.batch_size = batch_size
        self.max_env_steps = max_env_steps
        self.random_sample_timestep = random_sample_timestep
        self.start_update_timestep = start_update_timestep
        self.update_interval = update_interval
        self.action_noise_scale = action_noise_scale
        if load_dir != "" and os.path.exists(load_dir):
            self.agent.load_model(load_dir)

    def train(self):
        train_traj_returns = [0]
        train_traj_lengths = [0]
        traj_return = 0
        traj_length = 0
        done = False
        obs = self.train_env.reset()
        for env_step in trange(self.max_env_steps): # if system is windows, add ascii=True to tqdm parameters to avoid powershell bugs
            self.pre_iter()
            log_infos = {}
            
            if env_step < self.random_sample_timestep:
                action = self.train_env.action_space.sample()
            else:
                action = self.agent.select_action(obs, deterministic=True)['action']
                # add noise and clip action
                action = action + np.random.normal(size = action.shape, scale=self.action_noise_scale)
                action = np.clip(action, self.action_lower_bound, self.action_upper_bound)
            next_obs, reward, done, _ = self.train_env.step(action)
            traj_length  += 1
            traj_return += reward
            self.buffer.add_transition(obs, action, next_obs, reward, done)
            obs = next_obs
            if done or traj_length >= self.max_trajectory_length:
                obs = self.train_env.reset()
                train_traj_returns.append(traj_return)
                train_traj_lengths.append(traj_length)
                traj_length = 0
                traj_return = 0
            log_infos["performance/train_return"] = train_traj_returns[-1]
            log_infos["performance/train_length"] =  train_traj_lengths[-1]

            if env_step > self.start_update_timestep and env_step % self.update_interval == 0:
                for i in range(self.update_interval):
                    data_batch = self.buffer.sample(self.batch_size)
                    loss_dict = self.agent.update(data_batch)

                log_infos.update(loss_dict)
           
            self.post_iter(log_infos, env_step)
    




            


