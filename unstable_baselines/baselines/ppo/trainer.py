from unstable_baselines.common.trainer import BaseTrainer
import numpy as np
from tqdm import trange
from operator import itemgetter
import os

class PPOTrainer(BaseTrainer):
    def __init__(self, agent, train_env, eval_env, rollout_buffer,
            batch_size,
            max_env_steps=1000,
            start_timestep=0,
            num_trajs_per_epoch=-1,
            max_env_steps_per_epoch=2000,
            gamma=0.99, 
            load_dir="",
            gae_lambda=0.8,
            **kwargs):
        super(PPOTrainer, self).__init__(agent, train_env, eval_env, **kwargs)
        self.agent = agent
        self.rollout_buffer = rollout_buffer
        self.env = env 
        self.eval_env = eval_env
        #hyperparameters
        self.batch_size = batch_size
        self.start_timestep = start_timestep
        self.gamma = gamma
        self.gae_lambda=gae_lambda
        self.max_epoch = int(np.ceil(max_env_steps / max_env_steps_per_epoch))
        if load_dir != "" and os.path.exists(load_dir):
            self.agent.load(load_dir)

    def train(self):
        train_traj_returns = []
        train_traj_lengths = []
        tot_env_steps = 0

        for epoch_id in trange(self.max_epoch): 
            self.pre_iter()
            log_infos = {}
            for env_step in range(self.num_env_steps_per_epoch):
                # get action
                action, log_prob = itemgetter("action", "log_prob")(self.agent.select_action(obs))
                next_obs, reward, done, _ = self.train_env.step(action)

                traj_return += reward
                traj_length += 1
                tot_env_steps += 1

                # save
                value = self.agent.estimate_value(obs)[0]
                self.buffer.store(obs, action, reward, value, log_prob)
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
    
            #update network
            data_batch = self.rollout_buffer.sample(self.batch_size)
            loss_dict = self.agent.update(data_batch)
            log_infos.update(loss_dict)

            
            self.post_iter(log_infos, tot_env_steps)




