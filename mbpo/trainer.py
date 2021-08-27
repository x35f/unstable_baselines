from common.util import hard_update_network, soft_update_network, second_to_time_str
from common.trainer import BaseTrainer
import numpy as np
from tqdm import tqdm
from time import time
import cv2
import os
from tqdm import  tqdm
import torch
class MBPOTrainer(BaseTrainer):
    def __init__(self, agent, env, eval_env, env_buffer, model_buffer, rollout_step_generator, logger, 
            batch_size=32,
            rollout_batch_size=32, 
            num_updates_per_epoch=20, # G
            num_model_rollouts=100, #  M
            max_trajectory_length=1000,
            test_interval=10,
            num_test_trajectories=5,
            max_epoch=100000,
            save_model_interval=10000,
            start_timestep=1000,
            save_video_demo_interval=10000,
            num_steps_per_iteration=1,
            log_interval=100,
            load_dir="",
            **kwargs):
        self.agent = agent
        self.env_buffer = env_buffer
        self.model_buffer = model_buffer
        self.logger = logger
        self.env = env 
        self.eval_env = eval_env
        self.rollout_step_generator = rollout_step_generator
        #hyperparameters
        self.num_steps_per_iteration = num_steps_per_iteration
        self.num_model_rollouts = num_model_rollouts 
        self.batch_size = batch_size
        self.rollout_batch_size = rollout_batch_size
        self.num_updates_per_epoch = num_updates_per_epoch
        self.max_trajectory_length = max_trajectory_length
        self.test_interval = test_interval
        self.num_test_trajectories = num_test_trajectories
        self.max_epoch = max_epoch
        self.save_model_interval = save_model_interval
        self.save_video_demo_interval = save_video_demo_interval
        self.start_timestep = start_timestep
        self.log_interval = log_interval
        if load_dir != "" and os.path.exists(load_dir):
            self.agent.load(load_dir)

    def warmup(self, warmup_steps=2000):
        #add warmup transitions to buffer
        for step in range(self.num_steps_per_iteration):
            action, _ = self.agent.select_action(state)
            next_state, reward, done, _ = self.env.step(action)
            traj_length  += 1
            traj_reward += reward
            if traj_length >= self.max_trajectory_length - 1:
                done = True
            if self.agent.per:
                self.env_buffer.add_tuple(state, action, next_state, reward, float(done), self.buffer.max)
            else:
                self.env_buffer.add_tuple(state, action, next_state, reward, float(done))
            state = next_state
            if done or traj_length >= self.max_trajectory_length - 1:
                state = self.env.reset()
                traj_length = 0
                traj_reward = 0

    def train(self):
        train_traj_rewards = [0]
        train_traj_lengths = [0]
        iteration_durations = []
        self.warmup(self.start_timestep)
        tot_env_steps = self.start_timestep
        state = self.env.reset()
        traj_reward = 0
        traj_length = 0
        done = False
        state = self.env.reset()
        for epoch in tqdm(range(self.max_epoch)): # if system is windows, add ascii=True to tqdm parameters to avoid powershell bugs
            iteration_start_time = time()


            rollout_step = self.rollout_step_generator.next()

            #train model on env_buffer via maximum likelihood
            data_batch = self.env_buffer.sample_batch(self.batch_size)
            model_loss_dict = self.agent.train_model(data_batch)
            #for e steps do
            for step in range(self.num_steps_per_iteration):
                #take action in environment according to \pi, add to D_env
                action, _ = self.agent.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                traj_length  += 1
                traj_reward += reward
                if traj_length >= self.max_trajectory_length - 1:
                    done = True
                self.env_buffer.add_tuple(state, action, next_state, reward, float(done))
                state = next_state
                if done or traj_length >= self.max_trajectory_length - 1:
                    state = self.env.reset()
                    train_traj_rewards.append(traj_reward / self.env.reward_scale)
                    train_traj_lengths.append(traj_length)
                    self.logger.log_var("return/train",traj_reward / self.env.reward_scale, tot_env_steps)
                    self.logger.log_var("length/train",traj_length, tot_env_steps)
                    traj_length = 0
                    traj_reward = 0
                tot_env_steps += 1

                #for m model rollouts do
                for rollout_step in range(self.num_model_rollouts):
                    #sample s_t uniformly from D_env
                    data_batch = self.env_buffer.sample(self.rollout_batch_size)
                    #perform k-step model rollout starting from s_t using policy\pi
                    generated_transitions = self.agent.rollout(data_batch, rollout_step)
                    #add the transitions to D_model
                    for s, a, ns, r, d in zip(generated_transitions['state'], \
                                                generated_transitions['action'], \
                                                generated_transitions['next_state'], \
                                                generated_transitions['reward'], \
                                                generated_transitions['done']):
                        self.model_buffer.add_transition(s, a, ns, r, d)

                #for G gradient updates do
                for update_step in range(self.num_updates_per_epoch):
                    data_batch = self.model_buffer.sample_batch(self.batch_size)
                    policy_loss_dict = self.agent.update(data_batch)
           
            iteration_end_time = time()
            iteration_duration = iteration_end_time - iteration_start_time
            iteration_durations.append(iteration_duration)
            if ite % self.log_interval == 0:
                for loss_name in model_loss_dict:
                    self.logger.log_var(loss_name, loss_dict[loss_name], tot_env_steps)
                for loss_name in policy_loss_dict:
                    self.logger.log_var(loss_name, loss_dict[loss_name], tot_env_steps)
            if ite % self.test_interval == 0:
                log_dict = self.test()
                avg_test_reward = log_dict['return/test']
                for log_key in log_dict:
                    self.logger.log_var(log_key, log_dict[log_key], tot_env_steps)
                remaining_seconds = int((self.max_iteration - ite + 1) * np.mean(iteration_durations[-100:]))
                time_remaining_str = second_to_time_str(remaining_seconds)
                summary_str = "iteration {}/{}:\ttrain return {:.02f}\ttest return {:02f}\teta: {}".format(ite, self.max_iteration, train_traj_rewards[-1],avg_test_reward,time_remaining_str)
                self.logger.log_str(summary_str)
            if ite % self.save_model_interval == 0:
                self.agent.save_model(self.logger.log_dir, ite)
            if ite % self.save_video_demo_interval == 0:
                self.save_video_demo(ite)


    def test(self):
        #print("\033[32m -------------------testing----------------------\033[0m")
        rewards = []
        lengths = []
        for episode in range(self.num_test_trajectories):
            traj_reward = 0
            traj_length = 0
            state = self.eval_env.reset()
            for step in range(self.max_trajectory_length):
                action, _ = self.agent.select_action(state, evaluate=True)
                next_state, reward, done, _ = self.eval_env.step(action)
                traj_reward += reward
                state = next_state
                traj_length += 1 
                if done:
                    break
            lengths.append(traj_length)
            traj_reward /= self.eval_env.reward_scale
            rewards.append(traj_reward)
        return {
            "return/test": np.mean(rewards),
            "length/test": np.mean(lengths)
        }

    def save_video_demo(self, ite, width=128, height=128, fps=30):
        video_demo_dir = os.path.join(self.logger.log_dir,"demos")
        if not os.path.exists(video_demo_dir):
            os.makedirs(video_demo_dir)
        video_size = (height, width)
        video_save_path = os.path.join(video_demo_dir, "ite_{}.avi".format(ite))

        #initilialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(video_save_path, fourcc, fps, video_size)

        #rollout to generate pictures and write video
        state = self.eval_env.reset()
        img = self.eval_env.render(mode="rgb_array", width=width, height=height)
        traj_imgs =[img.astype(np.uint8)]
        for step in range(self.max_trajectory_length):
            action, _ = self.agent.select_action(state, evaluate=True)
            next_state, reward, done, _ = self.eval_env.step(action)
            state = next_state
            img = self.eval_env.render(mode="rgb_array", width=width, height=height)
            video_writer.write(img)
            if done:
                break
                
        video_writer.release()




            


