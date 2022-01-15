from unstable_baselines.common.trainer import BaseTrainer
import numpy as np
from tqdm import trange
import os
from time import time


class REDQTrainer(BaseTrainer):
    def __init__(self, agent, train_env, eval_env, buffer, 
            batch_size,
            max_env_steps,
            warmup_timesteps,
            update_policy_interval,
            utd,
            load_dir="",
            **kwargs):
        super(REDQTrainer, self).__init__(agent, train_env, eval_env, **kwargs)
        self.agent = agent
        self.buffer = buffer
        self.train_env = train_env 
        self.eval_env = eval_env
        #hyperparameters
        self.batch_size = batch_size
        self.max_env_steps = max_env_steps
        self.warmup_timesteps = warmup_timesteps
        self.update_policy_interval=update_policy_interval
        self.utd = utd
        if load_dir != "" and os.path.exists(load_dir):
            self.agent.load(load_dir)

    def warmup(self):
        obs = self.train_env.reset()
        for step in trange(self.warmup_timesteps):
            action = self.train_env.action_space.sample()
            next_obs, reward, done, info = self.train_env.step(action)
            self.buffer.add_transition(obs, action, next_obs, reward, float(done))
            obs = next_obs
            if done:
                obs = self.train_env.reset()

    def train(self, update_policy=False):
        train_traj_returns = [0]
        train_traj_lengths = [0]
        done = False
        self.warmup()
        tot_env_steps = self.warmup_timesteps
        tot_update_steps = 0
        obs = self.train_env.reset()
        traj_return = 0
        traj_length = 0
        for env_step in trange(self.max_env_steps - self.warmup_timesteps):
            self.pre_iter()
            log_infos = {}
            
            #interact with environment and add to buffer
            action = self.agent.select_action(obs)['action']
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

            #update agent
            train_agent_start_time = time()
            for train_step in range(self.utd):
                data_batch = self.buffer.sample(self.batch_size)
                update_policy = tot_update_steps % self.update_policy_interval == 0
                train_agent_log_infos = self.agent.update(data_batch, update_policy=update_policy) 
                tot_update_steps += 1
                log_infos.update(train_agent_log_infos)
            train_agent_used_time = time() - train_agent_start_time

            log_infos['times/train_agent'] = train_agent_used_time

            self.post_iter(log_infos, tot_env_steps)