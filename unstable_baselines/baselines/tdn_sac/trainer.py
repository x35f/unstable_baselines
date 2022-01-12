from unstable_baselines.common.util import second_to_time_str
from unstable_baselines.common.trainer import BaseTrainer
import numpy as np
from tqdm import tqdm
from time import time
import cv2
import os
from tqdm import  tqdm

class TDNSACTrainer(BaseTrainer):
    def __init__(self, agent, env, eval_env, buffer, logger, 
            batch_size=32,
            num_updates_per_iteration=20,
            max_trajectory_length=1000,
            test_interval=10,
            num_test_trajectories=5,
            max_iteration=100000,
            save_model_interval=10000,
            start_timestep=1000,
            save_video_demo_interval=10000,
            num_steps_per_iteration=1,
            log_interval=100,
            update_n_interval=10,
            n=3,
            adaptive_config=None,
            load_dir="",
            **kwargs):
        self.agent = agent
        self.buffer = buffer
        self.logger = logger
        self.env = env 
        self.eval_env = eval_env
        #hyperparameters
        self.num_steps_per_iteration = num_steps_per_iteration
        self.batch_size = batch_size
        self.num_updates_per_ite = num_updates_per_iteration
        self.max_trajectory_length = max_trajectory_length
        self.test_interval = test_interval
        self.num_test_trajectories = num_test_trajectories
        self.max_iteration = max_iteration
        self.save_model_interval = save_model_interval
        self.save_video_demo_interval = save_video_demo_interval
        self.start_timestep = start_timestep
        self.log_interval = log_interval
        self.update_n_interval = update_n_interval
        self.n = n
        self.adaptive_config = adaptive_config
        if load_dir != "" and os.path.exists(load_dir):
            self.agent.load(load_dir)

    def train(self):
        train_traj_rewards = [0]
        train_traj_lengths = [0]
        iteration_durations = []
        tot_env_steps = 0
        state = self.env.reset()
        traj_reward = 0
        traj_length = 0
        done = False
        state = self.env.reset()
        for ite in tqdm(range(self.max_iteration)): # if system is windows, add ascii=True to tqdm parameters to avoid powershell bugs
            iteration_start_time = time()
            for step in range(self.num_steps_per_iteration):
                action, _ = self.agent.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                traj_length  += 1
                traj_reward += reward
                if traj_length >= self.max_trajectory_length - 1:
                    done = True
                self.buffer.add_tuple(state, action, next_state, reward, float(done))
                state = next_state
                if done or traj_length >= self.max_trajectory_length - 1:
                    state = self.env.reset()
                    train_traj_rewards.append(traj_reward / self.env.reward_scale)
                    train_traj_lengths.append(traj_length)
                    self.logger.log_var("return/train",traj_reward / self.env.reward_scale, tot_env_steps)
                    self.logger.log_var("length/train",traj_length, tot_env_steps)
                    traj_length = 0
                    traj_reward = 0
                tot_env_steps += 1
            if tot_env_steps < self.start_timestep:
                continue

            if ite % self.update_n_interval == 0:
                self.update_n()
                self.logger.log_var("info/n",self.n, tot_env_steps)

            #update network
            for update in range(self.num_updates_per_ite):
                data_batch = self.buffer.sample(self.batch_size, step_size = self.n)
                loss_dict = self.agent.update(data_batch)
                  
            iteration_end_time = time()
            iteration_duration = iteration_end_time - iteration_start_time
            iteration_durations.append(iteration_duration)
            
            if ite % self.log_interval == 0:
                for loss_name in loss_dict:
                    self.logger.log_var(loss_name, loss_dict[loss_name], tot_env_steps)
            if ite % self.test_interval == 0:
                log_dict = self.test()
                avg_test_reward = log_dict['return/test']
                for log_key in log_dict:
                    self.logger.log_var(log_key, log_dict[log_key], tot_env_steps)
                remaining_seconds = int((self.max_iteration - ite + 1) * np.mean(iteration_durations[-100:]))
                time_remaining_str = second_to_time_str(remaining_seconds)
                summary_str = "iteration {}/{}:\ttrain return {:.02f}\ttest return {:02f}\teta: {}".format(ite, self.max_iteration, train_traj_rewards[-1],avg_test_reward,time_remaining_str)
                self.logger.log_str(summary_str)
            if ite % self.save_model_interval == 0:
                self.agent.save_model(self.logger.log_dir, ite)
            if ite % self.save_video_demo_interval == 0:
                self.save_video_demo(ite)

    def update_n(self, bellman_error):
        if self.adaptive_config is None:
            return
        return 
        n_choices = self.n_choices
        bias_estimator = adaptive_config['bias_estimator']
        variance_estimator = adaptive_config['bias_estimator']
        value_estimator = adaptive_config['bias_estimator']
        num_samples = adaptive_config['num_samples']
        info = {
            "buffer": self.buffer,
            "n": None,
            "bellman_error": None,
            "value_network": self.agent.value_network,
            "max_trajectory_length": self.max_trajectory_length,
            "num_samples": num_samples
        }
        estimates = []
        for n in n_choices:
            info['n'] = n
            data_batch = self.buffer.sample(self.batch_size, step_size = n)
            bellman_error = self.agent.estimate_bellman_error(n, data_batch)
            agent.estimate_bellman_error(n)
            bias = get_bias(bias_estimator, info)
            variance = get_variance(variance_estimator, info)
            sigma = np.sqrt(sigma)
            value_bound = get_value_bound(value_estimator, info)
            confidence_bound = Bernstein_confidence_bound(sigma, value_bound)
            curr_estimate = {
                "n": n,
                "bias": bias,
                "confidence_bound": confidence_bound,
                "num_samples": num_samples,
                "bellman_error": bellman_error
            }
            estimates.append(curr_estimate)
        selected_n = calculate_last_intersected(estimates)
        self.n = selected_n

    def test(self):
        rewards = []
        lengths = []
        for episode in range(self.num_test_trajectories):
            traj_reward = 0
            traj_length = 0
            state = self.eval_env.reset()
            for step in range(self.max_trajectory_length):
                action, _ = self.agent.select_action(state, deterministic=True)
                next_state, reward, done, _ = self.eval_env.step(action)
                traj_reward += reward
                state = next_state
                traj_length += 1 
                if done:
                    break
            lengths.append(traj_length)
            traj_reward /= self.eval_env.reward_scale
            rewards.append(traj_reward)
        return {
            "return/test": np.mean(rewards),
            "length/test": np.mean(lengths)
        }



            


