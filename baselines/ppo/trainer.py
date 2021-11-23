from common.util import  second_to_time_str
from common.trainer import BaseTrainer
import numpy as np
from tqdm import tqdm
from time import time
import cv2
import os
from tqdm import  tqdm
class PPOTrainer(BaseTrainer):
    def __init__(self, agent, env, eval_env, rollout_buffer, logger, 
            batch_size=64,
            max_trajectory_length=1000,
            test_interval=10,
            num_test_trajectories=5,
            max_iteration=1000,
            save_model_interval=50,
            start_timestep=0,
            save_video_demo_interval=10000,
            num_trajs_per_iteration=-1,
            max_steps_per_iteration=2000,
            log_interval=100,
            gamma=0.99, 
            epoch=10,
            load_dir="",
            gae_lambda=0.8,
            n=1,
            **kwargs):
        logger.log_str("Auxilary parameters for trainer: {}".format(kwargs))
        self.agent = agent
        self.rollout_buffer = rollout_buffer
        self.logger = logger
        self.env = env 
        self.eval_env = eval_env
        #hyperparameters
        self.num_trajs_per_iteration = num_trajs_per_iteration
        self.max_steps_per_iteration = max_steps_per_iteration
        self.batch_size = batch_size
        self.max_trajectory_length = max_trajectory_length
        self.test_interval = test_interval
        self.num_test_trajectories = num_test_trajectories
        self.max_iteration = max_iteration
        self.save_model_interval = save_model_interval
        self.save_video_demo_interval = save_video_demo_interval
        self.start_timestep = start_timestep
        self.log_interval = log_interval
        self.gamma = gamma
        self.gae_lambda=gae_lambda
        self.epoch = epoch
        self.n = n
        if load_dir != "" and os.path.exists(load_dir):
            self.agent.load(load_dir)

    def train(self):
        train_traj_rewards = []
        train_traj_lengths = []
        iteration_durations = []
        tot_env_steps = 0
        for ite in tqdm(range(self.max_iteration)): # if system is windows, add ascii=True to tqdm parameters to avoid powershell bugs
            iteration_start_time = time()
            train_traj_reward, train_traj_length = self.rollout_buffer.collect_trajectories(self.env, self.agent, self.agent.v_network, gae_lambda = self.gae_lambda)
            train_traj_rewards.append(train_traj_reward)
            train_traj_lengths.append(train_traj_length)
            tot_env_steps += self.rollout_buffer.size
            if tot_env_steps < self.start_timestep:
                continue

            num_updates = int( np.ceil(self.rollout_buffer.size / self.batch_size * self.epoch))
            #update network
            for update in range(num_updates):
                data_batch = self.rollout_buffer.sample_batch(self.batch_size)
                loss_dict = self.agent.update(data_batch)
            #reset rollout buffer
            self.rollout_buffer.reset()

            iteration_end_time = time()
            iteration_duration = iteration_end_time - iteration_start_time
            iteration_durations.append(iteration_duration)
            iteration_durations = iteration_durations[-100:]
            
            if ite % self.log_interval == 0:
                for loss_name in loss_dict:
                    self.logger.log_var(loss_name, loss_dict[loss_name], tot_env_steps)
                self.logger.log_var("return/train", train_traj_rewards[-1], tot_env_steps)
                self.logger.log_var("length/train", train_traj_lengths[-1], tot_env_steps)
            if ite % self.test_interval == 0:
                log_dict = self.test()
                avg_test_reward = log_dict['return/test']
                for log_key in log_dict:
                    self.logger.log_var(log_key, log_dict[log_key], tot_env_steps)
                remaining_seconds = int((self.max_iteration - ite + 1) * np.mean(iteration_durations))
                time_remaining_str = second_to_time_str(remaining_seconds)
                summary_str = "iteration {}/{}:\ttrain return {:.02f}\ttest return {:02f}\teta: {}".format(ite, self.max_iteration, train_traj_rewards[-1],avg_test_reward,time_remaining_str)
                self.logger.log_str(summary_str)
            if ite % self.save_model_interval == 0:
                self.agent.save_model(self.logger.log_dir, ite)
            if ite % self.save_video_demo_interval == 0:
                self.save_video_demo(ite)


    def test(self):
        rewards = []
        lengths = []
        for episode in range(self.num_test_trajectories):
            traj_reward = 0
            traj_length = 0
            state = self.eval_env.reset()
            for step in range(self.max_trajectory_length):
                action = self.agent.select_action(state, deterministic=True)
                action = np.clip(action, self.eval_env.action_space.low, self.eval_env.action_space.high)
                next_state, reward, done, _ = self.eval_env.step(action)
                traj_reward += reward
                state = next_state
                traj_length += 1 
                if done:
                    break
            lengths.append(traj_length)
            rewards.append(traj_reward)
        return {
            "return/test": np.mean(rewards),
            "length/test": np.mean(lengths)
        }



