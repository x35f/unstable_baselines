from unstable_baselines.common.util import second_to_time_str
from unstable_baselines.common.trainer import BaseTrainer
import numpy as np
from tqdm import trange
import os
from time import time
class REDQTrainer(BaseTrainer):
    def __init__(self, agent, train_env, eval_env, buffer, 
            batch_size,
            max_env_steps,
            start_timestep,
            load_dir="",
            **kwargs):
        super(REDQTrainer, self).__init__(agent, train_env, eval_env, **kwargs)
        self.agent = agent
        self.buffer = buffer
        self.train_env = train_env 
        self.eval_env = eval_env
        #hyperparameters
        self.batch_size = batch_size
        self.max_env_steps = max_env_steps
        self.start_timestep = start_timestep
        if load_dir != "" and os.path.exists(load_dir):
            self.agent.load(load_dir)

    def train(self):
        tot_env_steps = 0
        train_traj_returns = [0]
        train_traj_lengths = [0]
        done = False
        state = self.train_env.reset()
        traj_return = 0
        traj_length = 0
        for env_step in trange(self.max_env_steps):
            self.pre_iter()
            log_infos = {}
            
            #interact with environment and add to buffer
            action = self.agent.select_action(state)['action']
            next_state, reward, done, _ = self.train_env.step(action)
            tot_env_steps += 1
            traj_length += 1
            traj_return += reward
            self.buffer.add_transition(state, action, next_state, reward, float(done))
            state = next_state
            if done or traj_length >= self.max_trajectory_length - 1:
                state = self.train_env.reset()
                train_traj_returns.append(traj_return)
                train_traj_lengths.append(traj_length)
                traj_length = 0
                traj_return = 0
            log_infos['performance/train_return'] = train_traj_returns[-1]
            log_infos['performance/train_length'] = train_traj_lengths[-1]
            if env_step < self.start_timestep:
                continue

            #update agent
            train_agent_start_time = time()
            data_batch = self.buffer.sample(self.batch_size)
            train_agent_log_infos = self.agent.update(data_batch)
            train_agent_used_time = time() - train_agent_start_time

            log_infos['times/train_agent'] = train_agent_used_time
            log_infos.update(train_agent_log_infos)

            self.post_iter(log_infos, tot_env_steps)