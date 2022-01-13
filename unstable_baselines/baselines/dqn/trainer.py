import os
from unstable_baselines.common import util
import cv2
from unstable_baselines.common.trainer import BaseTrainer
from tqdm import trange
import random

class DQNTrainer(BaseTrainer):
    def __init__(self, agent, train_env, eval_env, buffer, 
            batch_size=32,
            num_updates_per_epoch=500,
            num_env_steps_per_epoch = 500,
            max_epoch=100000,
            epsilon=0.9,
            start_timestep=1000,
            **kwargs):
        super(DQNTrainer, self).__init__(agent, train_env, eval_env, **kwargs)
        self.buffer = buffer
        self.train_env = train_env 
        self.eval_env = eval_env
        #hyperparameters
        self.batch_size = batch_size
        self.num_updates_per_epoch = num_updates_per_epoch
        self.num_env_steps_per_epoch = num_env_steps_per_epoch
        self.max_epoch = max_epoch
        self.epsilon = epsilon
        self.start_timestep = start_timestep


    def train(self):
    
        train_traj_returns = [0]
        train_traj_lengths = [0]
        tot_env_steps = 0
        traj_return = 0
        traj_length = 0
        obs = self.train_env.reset()
        done = False
        tot_env_steps = 0
        for epoch in trange(self.max_epoch):
            self.pre_iter()
            log_infos = {}

            for env_step in range(self.num_env_steps_per_epoch):
                if random.random() < self.epsilon:
                    action = self.train_env.action_space.sample()
                else: 
                    action = self.agent.select_action(obs)['action']
                next_obs, reward, done, info = self.train_env.step(action)
                tot_env_steps += 1
                traj_length += 1
                traj_return += reward
                self.buffer.add_transition(obs, action, next_obs, reward, float(done))
                obs = next_obs
                if done or traj_length >= self.max_trajectory_length:
                    obs = self.train_env.reset()
                    train_traj_returns.append(traj_return)
                    train_traj_lengths.append(traj_length)
                    traj_length = 0
                    traj_return = 0
                tot_env_steps += 1
                log_infos["performance/train_return"] = train_traj_returns[-1]
                log_infos["performance/train_length"] =  train_traj_lengths[-1]
            if tot_env_steps < self.start_timestep:
                continue

            for update in range(self.num_updates_per_epoch):
                data_batch = self.buffer.sample(self.batch_size)
                train_agent_log_info = self.agent.update(data_batch)
            log_infos.update(train_agent_log_info)

            self.post_iter(log_infos, tot_env_steps)

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
        img = self.eval_env.render(mode="rgb_array")
        video_writer.write(img)
        for step in range(self.max_trajectory_length):
            action = self.agent.select_action(state)['action']
            next_state, reward, done, _ = self.eval_env.step(action)
            state = next_state
            img = self.eval_env.render(mode="rgb_array")
            video_writer.write(img)
            if done:
                break
                
        video_writer.release()