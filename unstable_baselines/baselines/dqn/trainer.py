from unstable_baselines.common.util import second_to_time_str
from unstable_baselines.common.trainer import BaseTrainer
from unstable_baselines.common import util
import numpy as np
from time import time
import random
import os
import cv2

class DQNTrainer(BaseTrainer):
    def __init__(self, agent, env, eval_env, buffer, logger, 
            batch_size=32,
            num_updates_per_iteration=500,
            num_steps_per_iteration = 500,
            max_trajectory_length=500,
            test_interval=10,
            log_interval=100,
            num_test_trajectories=5,
            max_iteration=100000,
            epsilon=0.9,
            save_video_demo_interval=10000,
            save_model_interval=10000,
            start_timestep=1000,
            **kwargs):
        self.agent = agent
        self.buffer = buffer
        self.logger = logger
        self.env = env 
        self.eval_env = eval_env
        #hyperparameters
        self.batch_size = batch_size
        self.num_updates_per_ite = num_updates_per_iteration
        self.num_steps_per_iteration = num_steps_per_iteration
        self.max_trajectory_length = max_trajectory_length
        self.test_interval = test_interval
        self.num_test_trajectories = num_test_trajectories
        self.max_iteration = max_iteration
        self.epsilon = epsilon
        self.save_video_demo_interval = save_video_demo_interval
        self.save_model_interval = save_model_interval
        self.log_interval = log_interval
        self.start_timestep = start_timestep


    def train(self):
    
        train_traj_rewards = []
        train_traj_lengths = []
        iteration_durations = []
        tot_env_steps = 0
        state = self.env.reset()
        traj_reward = 0
        traj_length = 0
        done = False
        state = self.env.reset()
        for ite in range(self.max_iteration):
            iteration_start_time = time()
            for step in range(self.num_steps_per_iteration):
                if random.random() < self.epsilon:
                    action = self.env.action_space.sample()
                else: 
                    action = self.agent.select_action(state)
                next_state, reward, done, info = self.env.step(action)
                traj_length += 1
                traj_reward += reward
                self.buffer.add_tuple(state, action, next_state, reward, float(done))
                state = next_state
                if done or traj_length >= self.max_trajectory_length:
                    state = self.env.reset()
                    train_traj_rewards.append(traj_reward)
                    train_traj_lengths.append(traj_length)
                    self.logger.log_var("return/train",traj_reward, tot_env_steps)
                    self.logger.log_var("length/train",traj_length, tot_env_steps)
                    traj_length = 0
                    traj_reward = 0
                tot_env_steps += 1
            if tot_env_steps < self.start_timestep:
                continue

            for update in range(self.num_updates_per_ite):
                data_batch = self.buffer.sample(self.batch_size)
                loss_dict = self.agent.update(data_batch)

            iteration_end_time = time()
            iteration_duration = iteration_end_time - iteration_start_time
            iteration_durations.append(iteration_duration)
            
            if self.log_interval > 0 and ite % self.log_interval == 0:
                for loss_name in loss_dict:
                    self.logger.log_var(loss_name, loss_dict[loss_name], tot_env_steps)
            if self.test_interval > 0 and ite % self.test_interval == 0:
                log_dict = self.test()
                avg_test_reward = log_dict['return/test']
                for log_key in log_dict:
                    self.logger.log_var(log_key, log_dict[log_key], tot_env_steps)
                remaining_seconds = int((self.max_iteration - ite + 1) * np.mean(iteration_durations[-100:]))
                time_remaining_str = second_to_time_str(remaining_seconds)
                summary_str = "iteration {}/{}:\ttrain return {:.02f}\ttest return {:02f}\teta: {}".format(ite, self.max_iteration, train_traj_rewards[-1],avg_test_reward,time_remaining_str)
                self.logger.log_str(summary_str)
            if self.save_model_interval > 0 and ite % self.save_model_interval == 0:
                self.agent.save_model(ite)
            if self.save_video_demo_interval > 0 and ite % self.save_video_demo_interval == 0:
                self.save_video_demo(ite)


    def test(self):
        rewards = []
        lengths = []
        for episode in range(self.num_test_trajectories):
            traj_reward = 0
            traj_length = 0
            state = self.eval_env.reset()
            for step in range(self.max_trajectory_length):
                action = self.agent.select_action(state)
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

    def save_video_demo(self, ite, fps=30):
        video_demo_dir = os.path.join(util.logger.log_dir,"demos")
        if not os.path.exists(video_demo_dir):
            os.makedirs(video_demo_dir)
        video_size = (64, 64)
        video_save_path = os.path.join(video_demo_dir, "ite_{}.avi".format(ite))

        #initilialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(video_save_path, fourcc, fps, video_size)

        #rollout to generate pictures and write video
        state = self.eval_env.reset()
        img = self.eval_env.render(mode="rgb_array")
        for step in range(self.max_trajectory_length):
            action, _ = self.agent.select_action(state)
            next_state, reward, done, _ = self.eval_env.step(action)
            state = next_state
            img = self.eval_env.render(mode="rgb_array")
            video_writer.write(img)
            if done:
                break
                
        video_writer.release()
