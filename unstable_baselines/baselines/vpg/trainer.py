import os
import warnings
from time import time

import torch
import numpy as np
from tqdm import trange

from unstable_baselines.common.trainer import BaseTrainer
from unstable_baselines.common.util import second_to_time_str
from operator import itemgetter

class VPGTrainer(BaseTrainer):
    """ Vanilla Policy Gradient Trainer

    BaseTrainer Args
    ----------------
    agent, env, eval_env, buffer, logger

    kwargs Args
    -----------
    max_env_steps: int, default: 2e6

    num_steps_per_epoch: int, default: 1000
    """
    def __init__(self, agent, train_env, eval_env, buffer,
                max_env_steps: int,
                num_env_steps_per_epoch: int,
                 **kwargs):
        super(VPGTrainer, self).__init__(agent, train_env, eval_env, **kwargs)
        self.agent = agent
        self.buffer = buffer
        # experiment parameters
        self.max_env_steps = max_env_steps
        self.num_env_steps_per_epoch = num_env_steps_per_epoch
        self.max_epoch = self.max_env_steps // self.num_env_steps_per_epoch

    def train(self):
        train_traj_returns = [0]
        train_traj_lengths = [0]

        tot_env_steps = 0
        traj_return = 0
        traj_length = 0

        obs, info = self.train_env.reset()
        done = False

        # if system is Windows, add ascii=True to tqdm parameters to avoid powershell bugs
        for epoch_id in trange(self.max_epoch):
            self.pre_iter()
            log_infos = {}
            for env_step in range(self.num_env_steps_per_epoch):
                # get action
                action = itemgetter("action")(self.agent.select_action(obs))[0]
                next_obs, reward, done, truncated, info  = self.train_env.step(action)

                traj_return += reward
                traj_length += 1
                tot_env_steps += 1

                # save
                self.buffer.add_transition(obs, action, next_obs, reward, done, truncated)
                obs = next_obs

                truncated = traj_length == self.max_trajectory_length or truncated

                if done or truncated:
                    # log
                    train_traj_returns.append(traj_return)
                    train_traj_lengths.append(traj_length)
                    # reset env and pointer
                    obs, info = self.train_env.reset()
                    traj_return = 0
                    traj_length = 0
                self.post_step(tot_env_steps)
    
            log_infos['performance/train_return'] = train_traj_returns[-1]
            log_infos['performance/train_length'] = train_traj_lengths[-1]
            
            # update
            data_batch = self.buffer.get_batch(np.arange(self.buffer.max_sample_size))
            train_agent_log_infos = self.agent.update(data_batch)
            log_infos.update(train_agent_log_infos)

            self.post_iter(log_infos, tot_env_steps)
