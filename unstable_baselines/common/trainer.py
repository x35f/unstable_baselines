import numpy as np
from abc import ABC, abstractmethod
import torch
import os
import cv2
from time import time
from unstable_baselines.common import util
from unstable_baselines.common import util
class BaseTrainer():
    def __init__(self, agent, train_env, eval_env, 
            max_trajectory_length,
            log_interval,
            eval_interval,
            num_eval_trajectories,
            save_video_demo_interval,
            snapshot_interval,
            **kwargs):
        self.agent = agent
        self.train_env = train_env
        self.eval_env = eval_env
        self.max_trajectory_length = max_trajectory_length
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.num_eval_trajectories = num_eval_trajectories
        self.save_video_demo_interval = save_video_demo_interval
        self.snapshot_interval = snapshot_interval
        pass

    @abstractmethod
    def train(self):
        #do training 
        pass
    def pre_iter(self):
        self.ite_start_time = time()
    
    def post_iter(self, log_info_dict, timestamp):
        if timestamp % self.log_interval == 0:
            for loss_name in log_info_dict:
                util.logger.log_var(loss_name, log_info_dict[loss_name], timestamp)
        if timestamp % self.eval_interval == 0:
            eval_start_time = time()
            log_dict = self.evaluate()
            eval_used_time = time() - eval_start_time
            avg_test_return = log_dict['performance/eval_return']
            for log_key in log_dict:
                util.logger.log_var(log_key, log_dict[log_key], timestamp)
            util.logger.log_var("times/eval", eval_used_time, timestamp)
            summary_str = "Timestamp:{}\tEvaluation return {:02f}".format(timestamp, avg_test_return)
            util.logger.log_str(summary_str)
        if timestamp % self.snapshot_interval == 0:
            self.agent.snapshot(timestamp)
        if self.save_video_demo_interval > 0 and timestamp % self.save_video_demo_interval == 0:
            self.save_video_demo(timestamp)

    @torch.no_grad()
    def evaluate(self):
        traj_returns = []
        traj_lengths = []
        for traj_id in range(self.num_eval_trajectories):
            traj_return = 0
            traj_length = 0
            state = self.eval_env.reset()
            for step in range(self.max_trajectory_length):
                action = self.agent.select_action(state, deterministic=True)['action']
                next_state, reward, done, _ = self.eval_env.step(action)
                traj_return += reward
                state = next_state
                traj_length += 1 
                if done:
                    break
            traj_lengths.append(traj_length)
            traj_returns.append(traj_return)
        return {
            "performance/eval_return": np.mean(traj_returns),
            "performance/eval_length": np.mean(traj_lengths)
        }
        
        
    def save_video_demo(self, ite, width=256, height=256, fps=30):
        video_demo_dir = os.path.join(util.logger.log_dir,"demos")
        if not os.path.exists(video_demo_dir):
            os.makedirs(video_demo_dir)
        video_size = (height, width)
        video_save_path = os.path.join(video_demo_dir, "ite_{}.mp4".format(ite))

        #initilialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_save_path, fourcc, fps, video_size)

        #rollout to generate pictures and write video
        state = self.eval_env.reset()
        img = self.eval_env.render(mode="rgb_array", width=width, height=height)
        video_writer.write(img)
        for step in range(self.max_trajectory_length):
            action = self.agent.select_action(state)['action']
            next_state, reward, done, _ = self.eval_env.step(action)
            state = next_state
            img = self.eval_env.render(mode="rgb_array", width=width, height=height)
            video_writer.write(img)
            if done:
                break
                
        video_writer.release()