import os
import warnings
from time import time

import torch
import numpy as np
from tqdm import tqdm

from unstable_baselines.common.trainer import BaseTrainer
from unstable_baselines.common.util import second_to_time_str


class VPGTrainer(BaseTrainer):
    """ Vanilla Policy Gradient Trainer

    BaseTrainer Args
    ----------------
    agent, env, eval_env, buffer, logger

    kwargs Args
    -----------
    max_total_steps: int, default: 2e6

    max_trajectory_length: int, default: 1000, unit: step

    num_steps_per_iteration: int, default: 1000

    num_test_trajectories: int, default: 5, unit: trajectory

    log_interval: int, default: 2000, unit: step

    test_interval: int, default: 2000, unit: step

    save_model_interval: int, default: 5e5, unit: step

    save_video_demo_interval: int, default: 5e5, unit: step
    """
    def __init__(self, agent, env, eval_env, buffer, logger,
                #  max_total_steps: int=2e6,
                #  max_trajectory_length: int=1000,
                #  num_steps_per_iteration: int=1000,
                #  num_test_trajectories: int=5,
                #  log_interval: int=100,
                #  test_interval: int=100,
                #  save_model_interval: int=500000,
                #  save_video_demo_interval: int=500000,
                 **kwargs):
        self.agent = agent
        self.env = env
        self.eval_env = eval_env
        self.buffer = buffer
        self.logger = logger
        # experiment parameters
        self.max_total_steps = kwargs['max_total_steps']
        self.max_trajectory_length = kwargs['max_trajectory_length']
        self.num_steps_per_iteration = kwargs['num_steps_per_iteration']
        self.max_iteration = int(self.max_total_steps / self.num_steps_per_iteration)
        self.num_test_trajectories = kwargs['num_test_trajectories']
        self.log_interval = kwargs['log_interval']
        self.test_interval = kwargs['test_interval']
        self.save_model_interval = kwargs['save_model_interval']
        self.save_video_demo_interval = kwargs['save_video_demo_interval']

    def train(self):
        train_traj_rewards = [0]
        train_traj_lengths = [0]
        iteration_durations = []

        tot_env_steps = 0
        traj_reward = 0
        traj_length = 0

        done = False
        state = self.env.reset()

        # if system is Windows, add ascii=True to tqdm parameters to avoid powershell bugs
        for ite in tqdm(range(self.max_iteration)):
            iteration_start_time = time()
            for step in range(self.num_steps_per_iteration):
                # get action
                action, log_prob = self.agent.select_action(state)
                next_state, reward, done, _ = self.env.step(action)

                traj_reward += reward
                traj_length += 1
                tot_env_steps += 1

                # save
                value = self.agent.estimate_value(state)
                self.buffer.store(state, action, reward, value, log_prob)
                state = next_state

                timeout = traj_length == self.max_trajectory_length
                terminal = done or timeout
                iteration_ended = step == self.num_steps_per_iteration - 1
                if terminal or iteration_ended:
                    if timeout or iteration_ended:
                        # bootstrap
                        last_v = self.agent.estimate_value(state)
                    else:
                        last_v = 0
                    self.buffer.finish_path(last_v)
                    # log
                    train_traj_rewards.append(traj_reward)
                    train_traj_lengths.append(traj_length)
                    self.logger.log_var("return/train", traj_reward, tot_env_steps)
                    self.logger.log_var("length/train", traj_length, tot_env_steps)
                    # reset env and pointer
                    state = self.env.reset()
                    traj_reward = 0
                    traj_length = 0
            
            # update
            data_batch = self.buffer.get()
            loss_dict = self.agent.update(data_batch)

            iteration_end_time = time()
            iteration_duration = iteration_end_time - iteration_start_time
            iteration_durations.append(iteration_duration)
            # log tensorboard
            if tot_env_steps % self.log_interval == 0:
                for loss_name, loss_value in loss_dict.items():
                    self.logger.log_var(loss_name, loss_value, tot_env_steps)
            # evaluate policy
            if tot_env_steps % self.test_interval == 0:
                # evaluate
                log_dict = self.test()
                for name, log_value in log_dict.items():
                    self.logger.log_var(name, log_value, tot_env_steps)
                # calculate experiment time
                avg_test_reward = log_dict['return/test']
                remaining_seconds = int((self.max_iteration - ite + 1) * np.mean(iteration_durations[-100:]))
                time_remaining_str = second_to_time_str(remaining_seconds)
                self.logger.log_str(f"iteration {ite}/{self.max_iteration}\t"
                                    f"train return: {train_traj_rewards[-1]:.2f}\t"
                                    f"test return: {avg_test_reward:.2f}\t"
                                    f"eta: {time_remaining_str}")
            # save model
            if tot_env_steps % self.save_model_interval == 0:
                # util.debug_print(self.logger.log_dir, ite)
                self.agent.save_model(tot_env_steps)
            # save vedio demo
            if tot_env_steps % self.save_video_demo_interval == 0:
                self.save_video_demo(tot_env_steps)
                
    @torch.no_grad()
    def test(self):
        """ Evaluate stage.
        """
        rewards = []
        lengths = []
        for episode in range(self.num_test_trajectories):
            traj_reward = 0
            traj_length = 0
            state = self.eval_env.reset()
            for step in range(self.max_trajectory_length):
                action, _ = self.agent.select_action(state)     # TODO(mimeku): 是否要加一个deterministic
                next_state, reward, done, info = self.eval_env.step(action)
                traj_reward += reward
                traj_length += 1
                state = next_state
                if done:
                    break
            rewards.append(traj_reward)
            lengths.append(traj_length)
        return {
            "return/test": np.mean(rewards),
            "length/test": np.mean(lengths)
        }
