from unstable_baselines.common.trainer import BaseTrainer
import numpy as np
from tqdm import trange
from operator import itemgetter
import os
from time import time

class TRPOTrainer(BaseTrainer):
    def __init__(self, agent, train_env, eval_env, buffer,
            max_env_steps,
            num_env_steps_per_epoch,
            load_dir="",
            **kwargs):
        super(TRPOTrainer, self).__init__(agent, train_env, eval_env, **kwargs)
        self.agent = agent
        self.buffer = buffer
        #hyperparameters
        self.max_epoch = int(np.ceil(max_env_steps / num_env_steps_per_epoch))
        self.num_env_steps_per_epoch = num_env_steps_per_epoch
        if load_dir != "" and os.path.exists(load_dir):
            self.agent.load_snapshot(load_dir)

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

            self.buffer.clear()
            sample_start_time = time()
            env_steps = 0
            num_sampled_trajs = 0
            while env_steps < self.num_env_steps_per_epoch:
                obs = self.train_env.reset()
                traj_return = 0
                traj_length = 0
                for step in range(self.max_trajectory_length ):
                    # get action
                    action = itemgetter("action")(self.agent.select_action(obs, deterministic=False))
                    next_obs, reward, done, _ = self.train_env.step(action)

                    traj_return += reward
                    traj_length += 1
                    tot_env_steps += 1
                    env_steps += 1

                    # save
                    self.buffer.add_transition(obs, action, next_obs, reward, done)
                    obs = next_obs

                    timeout = traj_length == self.max_trajectory_length
                    terminal = done or timeout
                    if terminal:
                        # log
                        train_traj_returns.append(traj_return)
                        train_traj_lengths.append(traj_length)
                        # reset env and pointer
                        break
                num_sampled_trajs += 1
            sample_used_time = time() - sample_start_time
            log_infos['times/sample'] = sample_used_time
            log_infos["performance/train_return"] = np.mean(train_traj_returns[-num_sampled_trajs:])
            log_infos["performance/train_length"] =  np.mean(train_traj_lengths[-num_sampled_trajs:])
    
            #train agent
            train_agent_start_time = time()
            data_batch = self.buffer.get_batch(np.arange(self.buffer.max_sample_size))
            loss_dict = self.agent.update(data_batch)
            train_agent_used_time = time() - train_agent_start_time
            log_infos['times/train_agent'] = train_agent_used_time
            log_infos.update(loss_dict)


            self.eval_env.__setstate__(self.train_env.__getstate__())
            self.post_iter(log_infos, tot_env_steps)




