from unstable_baselines.common import util
from unstable_baselines.common.trainer import BaseTrainer
import numpy as np
from tqdm import tqdm
from time import time
import os
from tqdm import  tqdm

import warnings

class DDPGTrainer(BaseTrainer):
    def __init__(self, agent, env, eval_env, buffer,
            batch_size=32,
            max_trajectory_length=1000,
            eval_interval=10,
            num_eval_trajectories=5,
            max_iteration=100000,
            save_model_interval=10000,
            random_sample_timestep=10000,
            start_update_timestep=2000,
            update_interval=50,
            save_video_demo_interval=10000,
            log_interval=100,
            load_dir="",
            action_noise_scale = 0.1,
            **kwargs):
        warnings.warn('redundant arguments for trainer: {}'.format(kwargs))
        self.agent = agent
        self.buffer = buffer
        self.env = env 
        self.eval_env = eval_env
        #hyperparameters
        self.action_upper_bound  = env.action_space.high[0]
        self.action_lower_bound  = env.action_space.low[0]
        self.batch_size = batch_size
        self.max_trajectory_length = max_trajectory_length
        self.eval_interval = eval_interval
        self.num_eval_trajectories = num_eval_trajectories
        self.max_iteration = max_iteration
        self.save_model_interval = save_model_interval
        self.save_video_demo_interval = save_video_demo_interval
        self.random_sample_timestep = random_sample_timestep
        self.start_update_timestep = start_update_timestep
        self.update_interval = update_interval
        self.log_interval = log_interval
        self.action_noise_scale = action_noise_scale
        if load_dir != "" and os.path.exists(load_dir):
            self.agent.load(load_dir)

    def train(self):
        train_traj_rewards = [0]
        train_traj_lengths = [0]
        iteration_durations = []
        traj_reward = 0
        traj_length = 0
        done = False
        obs = self.env.reset()
        for ite in tqdm(range(self.max_iteration)): # if system is windows, add ascii=True to tqdm parameters to avoid powershell bugs
            iteration_start_time = time()
            
            if ite < self.random_sample_timestep:
                action = self.env.action_space.sample()
            else:
                action, _ = self.agent.select_action(obs)
                # add noise and clip action
                action = action + np.random.normal(size = action.shape, scale=self.action_noise_scale)
                action = np.clip(action, self.action_lower_bound, self.action_upper_bound)
            next_obs, reward, done, _ = self.env.step(action)
            traj_length  += 1
            traj_reward += reward
            if traj_length == self.max_trajectory_length:
                done = False # for mujoco env
            self.buffer.add_tuple(obs, action, next_obs, reward, done)
            obs = next_obs
            if done or traj_length >= self.max_trajectory_length:
                obs = self.env.reset()
                train_traj_rewards.append(traj_reward)
                train_traj_lengths.append(traj_length)
                traj_length = 0
                traj_reward = 0
            if ite < self.start_update_timestep or ite % self.update_interval != 0:
                continue
            
            for i in range(self.update_interval):
                data_batch = self.buffer.sample_batch(self.batch_size)
                loss_dict = self.agent.update(data_batch)
           
            iteration_end_time = time()
            iteration_duration = iteration_end_time - iteration_start_time
            iteration_durations.append(iteration_duration)

            if ite % self.log_interval == 0:
                util.logger.log_var("return/train",train_traj_rewards[-1], ite)
                util.logger.log_var("length/train",train_traj_lengths[-1], ite)
                for loss_name in loss_dict:
                    util.logger.log_var(loss_name, loss_dict[loss_name], ite)
                util.logger.log_var("time/train_iteration_duration(s)", np.mean(iteration_durations[-50:]), ite)

            if ite % self.eval_interval == 0:
                eval_start_time = time()
                log_dict = self.eval()
                eval_duration = time() - eval_start_time
                avg_eval_reward = log_dict['return/eval']
                for log_key in log_dict:
                    util.logger.log_var(log_key, log_dict[log_key], ite)
                util.logger.log_var('time/evaluation_duration(s)', eval_duration, ite)
                remaining_seconds = int((self.max_iteration - ite + 1) / self.update_interval * np.mean(iteration_durations[-50:]))
                time_remaining_str = util.second_to_time_str(remaining_seconds)
                summary_str = "iteration {}/{}:\ttrain return {:.02f}\teval return {:02f}\teta: {}".format(ite, self.max_iteration, train_traj_rewards[-1],avg_eval_reward,time_remaining_str)
                util.logger.log_str(summary_str)

            if ite % self.save_model_interval == 0:
                self.agent.save_model(ite)

            if ite % self.save_video_demo_interval == 0:
                self.save_video_demo(ite)


    def eval(self):
        rewards = []
        lengths = []
        for episode in range(self.num_eval_trajectories):
            traj_reward = 0
            traj_length = 0
            obs = self.eval_env.reset()
            for step in range(self.max_trajectory_length):
                action, _ = self.agent.select_action(obs)
                next_obs, reward, done, _ = self.eval_env.step(action)
                traj_reward += reward
                obs = next_obs
                traj_length += 1 
                if done:
                    break
            lengths.append(traj_length)
            traj_reward /= self.eval_env.reward_scale
            rewards.append(traj_reward)
        return {
            "return/eval": np.mean(rewards),
            "length/eval": np.mean(lengths)
        }

    




            


