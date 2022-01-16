from random import sample
from unstable_baselines.common.trainer import BaseTrainer
import numpy as np
from tqdm import trange
from operator import itemgetter
import os
from time import time
class PPOTrainer(BaseTrainer):
    def __init__(self, agent, train_env, eval_env, buffer,
            batch_size,
            max_env_steps,
            num_env_steps_per_epoch,
            load_dir="",
            **kwargs):
        super(PPOTrainer, self).__init__(agent, train_env, eval_env, **kwargs)
        self.agent = agent
        self.buffer = buffer
        #hyperparameters
        self.batch_size = batch_size
        self.max_epoch = int(np.ceil(max_env_steps / num_env_steps_per_epoch))
        self.num_env_steps_per_epoch = num_env_steps_per_epoch
        if load_dir != "" and os.path.exists(load_dir):
            self.agent.load(load_dir)

    def train(self):
        train_traj_returns = []
        train_traj_lengths = []
        tot_env_steps = 0
        traj_return = 0
        traj_length = 0
        obs = self.train_env.reset()
        for epoch_id in trange(self.max_epoch, colour='blue', desc='outer loop'): 
            self.pre_iter()
            log_infos = {}

            sample_start_time = time()
            for env_step in trange(self.num_env_steps_per_epoch, colour='green', desc='inner loop'):
                # get action
                action, log_prob = itemgetter("action", "log_prob")(self.agent.select_action(obs))
                next_obs, reward, done, _ = self.train_env.step(action)

                traj_return += reward
                traj_length += 1
                tot_env_steps += 1

                # save
                value = self.agent.estimate_value(obs)[0]
                self.buffer.add_transition(obs, action, reward, value, log_prob)
                obs = next_obs

                timeout = traj_length == self.max_trajectory_length
                terminal = done or timeout
                epoch_ended = env_step == self.num_env_steps_per_epoch - 1
                if terminal or epoch_ended:
                    if timeout or epoch_ended:
                        # bootstrap
                        last_v = self.agent.estimate_value(obs)
                    else:
                        last_v = 0
                    self.buffer.finish_path(last_v)
                    # log
                    train_traj_returns.append(traj_return)
                    train_traj_lengths.append(traj_length)
                    # reset env and pointer
                    obs = self.train_env.reset()
                    traj_return = 0
                    traj_length = 0
            sample_used_time = time() - sample_start_time
            log_infos['times/sample'] = sample_used_time
            log_infos["performance/train_return"] = train_traj_returns[-1]
            log_infos["performance/train_length"] =  train_traj_lengths[-1]
    
            #train agent
            train_agent_start_time = time()
            data_batch = self.buffer.get()
            loss_dict = self.agent.update(data_batch)
            train_agent_used_time = time() - train_agent_start_time
            log_infos['times/train_agent'] = train_agent_used_time
            log_infos.update(loss_dict)


            
            self.post_iter(log_infos, tot_env_steps)




