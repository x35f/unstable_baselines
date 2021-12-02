from unstable_baselines.common.util import second_to_time_str
from unstable_baselines.common.trainer import BaseTrainer
import numpy as np
from tqdm import tqdm
from time import time
import cv2
import os
import random
import torch
from unstable_baselines.common import util 

class PEARLTrainer(BaseTrainer):
    def __init__(self, agent, train_env, test_env, train_replay_buffers, train_encoder_buffers, test_buffer, logger, load_dir,
            batch_size=32,
            z_inference_batch_size=64,
            use_next_obs_in_context=False,
            max_iteration=500,
            num_tasks_per_gradient_update=16,
            num_train_tasks=50,
            num_test_tasks=10,
            num_train_tasks_per_iteration=5, 
            num_steps_prior=400, 
            num_steps_posterior=0, 
            num_extra_rl_steps_posterior=400,
            num_updates_per_iteration=2000,
            num_evaluations=2,
            adaptation_context_update_interval=1,
            num_adaptation_trajectories_before_posterior_sampling=1,
            num_steps_per_iteration=50,
            max_trajectory_length=200,
            test_interval=20,
            num_test_trajectories=5,
            save_model_interval=2000,
            start_timestep=2000,
            save_video_demo_interval=10000,
            log_interval=1,
            **kwargs):
        print("redundant arguments for trainer: {}".format(kwargs))
        self.agent = agent
        self.train_replay_buffers = train_replay_buffers
        self.train_encoder_buffers = train_encoder_buffers
        self.test_buffer = test_buffer
        self.logger = logger
        self.train_env = train_env 
        self.test_env = test_env
        #hyperparameters
        self.batch_size = batch_size
        self.z_inference_batch_size = z_inference_batch_size
        self.use_next_obs_in_context = use_next_obs_in_context
        self.num_train_tasks = num_train_tasks
        self.num_test_tasks = num_test_tasks
        self.max_iteration = max_iteration
        self.num_tasks_per_gradient_update = num_tasks_per_gradient_update
        self.num_train_tasks_per_iteration = num_train_tasks_per_iteration
        self.num_steps_prior = num_steps_prior
        self.num_steps_posterior = num_steps_posterior
        self.num_extra_rl_steps_posterior = num_extra_rl_steps_posterior
        self.num_updates_per_iteration = num_updates_per_iteration
        self.num_evaluations = num_evaluations
        self.adaptation_context_update_interval = adaptation_context_update_interval
        self.num_adaptation_trajectories_before_posterior_sampling = num_adaptation_trajectories_before_posterior_sampling
        self.num_steps_per_iteration = num_steps_per_iteration
        self.max_trajectory_length = max_trajectory_length
        self.test_interval = test_interval
        self.num_test_tasks = num_test_tasks
        self.num_test_trajectories = num_test_trajectories
        self.save_model_interval = save_model_interval
        self.start_timestep = start_timestep
        self.save_video_demo_interval = save_video_demo_interval
        self.log_interval = log_interval
        if load_dir != "":
            if  os.path.exists(load_dir):
                self.agent.load(load_dir)
            else:
                print("Load dir {} Not Found".format(load_dir))
                exit(0)

    def train(self):
        train_traj_rewards = [0]
        iteration_durations = []
        self.tot_env_steps = 0
        for ite in tqdm(range(self.max_iteration)): 
            train_task_returns = []
            iteration_start_time = time()
            if ite == 0: # collect initial pool of data
                for idx in range(self.num_train_tasks):
                    self.train_env.reset_task(idx)
                    initial_samples = self.collect_data(idx, self.train_env, self.start_timestep, 1, np.inf)
                    self.train_encoder_buffers[idx].add_traj(*initial_samples)
                    self.train_replay_buffers[idx].add_traj(*initial_samples)
                    self.tot_env_steps += len(initial_samples[0])

            #sample data from train_tasks
            train_task_indices = random.sample(range(self.num_train_tasks), self.num_train_tasks_per_iteration)
            for train_task_idx in train_task_indices:
                self.train_env.reset_task(train_task_idx)
                self.train_encoder_buffers[train_task_idx].clear()
                prior_samples, posterior_samples, extra_posterior_samples = [[]], [[]], [[]]
                # collect some trajectories with z ~ prior
                if self.num_steps_prior > 0:
                    self.agent.clear_z()
                    prior_samples = self.collect_data(train_task_idx, self.train_env, self.num_steps_prior, 1, np.inf)
                    self.train_encoder_buffers[train_task_idx].add_traj(*prior_samples)
                    self.train_replay_buffers[train_task_idx].add_traj(*prior_samples)

                # collect some trajectories with z ~ posterior
                if self.num_steps_posterior > 0:
                    self.agent.clear_z()
                    posterior_samples = self.collect_data(train_task_idx, self.train_env,self.num_steps_posterior, 1, self.adaptation_context_update_interval)
                    self.train_encoder_buffers[train_task_idx].add_traj(*posterior_samples)
                    self.train_replay_buffers[train_task_idx].add_traj(*posterior_samples)

                # trajectories from the policy handling z ~ posterior
                if self.num_extra_rl_steps_posterior > 0:
                    self.agent.clear_z()
                    extra_posterior_samples = self.collect_data(train_task_idx, self.train_env,self.num_extra_rl_steps_posterior, 1, self.adaptation_context_update_interval)
                    self.train_replay_buffers[train_task_idx].add_traj(*extra_posterior_samples)
                    #does not add to encoder buffer since it's extra posterior

                self.tot_env_steps += len(prior_samples[0]) + len(posterior_samples[0]) + len(extra_posterior_samples[0])
                train_task_returns.append(np.sum(extra_posterior_samples[3]))
            #perform training on batches of train tasks
            for train_step in range(self.num_updates_per_iteration):
                train_task_indices = np.random.choice(range(self.num_train_tasks), self.num_tasks_per_gradient_update)
                loss_dict = self.train_step(train_task_indices)
            train_traj_rewards.append(np.mean(train_task_returns))
            #post iteration logging
            iteration_end_time = time()
            iteration_duration = iteration_end_time - iteration_start_time
            iteration_durations.append(iteration_duration)
            if ite % self.log_interval == 0:
                self.logger.log_var('return/train', train_traj_rewards[-1], self.tot_env_steps)
                for loss_name in loss_dict:
                    self.logger.log_var(loss_name, loss_dict[loss_name], self.tot_env_steps)
            if ite % self.test_interval == 0:
                log_dict = self.test()
                avg_test_reward = log_dict['return/test']
                for log_key in log_dict:
                    self.logger.log_var(log_key, log_dict[log_key], self.tot_env_steps)
                remaining_seconds = int((self.max_iteration - ite + 1) * np.mean(iteration_durations[-100:]))
                time_remaining_str = second_to_time_str(remaining_seconds)
                summary_str = "iteration {}/{}:\ttrain return {:.02f}\ttest return {:02f}\teta: {}".format(ite, self.max_iteration, train_traj_rewards[-1],avg_test_reward,time_remaining_str)
                self.logger.log_str(summary_str)
                self.logger.log_var("time/train_iteration_duration", iteration_duration, self.tot_env_steps)
            if ite % self.save_model_interval == 0:
                self.agent.save_model(self.logger.log_dir, self.tot_env_steps)
            if ite % self.save_video_demo_interval == 0:
                self.save_video_demo(self.tot_env_steps)

    def train_step(self, train_task_indices):
        # sample train context
        context_batch = self.sample_context(train_task_indices, train=True)
        
        #sample data for sac update
        data_batch = [self.train_replay_buffers[index].sample_batch(batch_size=self.batch_size, to_tensor=False) for index in train_task_indices]
        #data_batch is of shape task_num, [obs, action, next_obs, rewards, done]
        data_batch = [[x[i] for x in data_batch] for i in range(len(data_batch[0]))]
        #data_batch is of shape [task_num, obs_dim], [task_num, action_dim], ...
        data_batch = [torch.FloatTensor(d).to(util.device) for d in data_batch]

        #clear z inference
        self.agent.clear_z(num_tasks=len(train_task_indices))
        
        #perform update on context and data
        loss_dict = self.agent.update(context_batch, data_batch)
        
        return loss_dict

    def sample_context(self, indices, train=True):
        if train:
            contexts = [list(self.train_replay_buffers[idx].sample_batch(self.z_inference_batch_size, to_tensor=False)) for idx in indices]
        else:
            contexts = [list(self.test_buffer.sample_batch(self.z_inference_batch_size, to_tensor=False)) for idx in indices]
        # task_num,  [states, actions, next_states, rewards, dones] of length batch_size
        # unsqueeze rewards for concatenation
        for i, task_context in enumerate(contexts):
            contexts[i][3] = np.expand_dims(contexts[i][3], axis=1)

        if self.use_next_obs_in_context:
            contexts = [task_context[:2] +  [task_context[2], task_context[3]]  for task_context in contexts]
        else:
            contexts = [task_context[:2] +  [task_context[3]] for task_context in contexts]
        # task_num,  [states, actions, rewards, (next_states)] of length batch_size
        contexts = [torch.cat([torch.FloatTensor(batch_context)for batch_context in task_context], dim=1) for task_context in contexts]
        contexts = torch.stack(contexts).to(util.device)
        #task_num, batch_size, context_dim
        return contexts

    def test(self):
        test_traj_returns = []
        
        self.agent.clear_z(num_tasks=1)
        for idx in range(self.num_test_tasks):
            #reset env and buffer
            self.test_env.reset_task(idx)
            self.test_buffer.clear()
            traj_returns =[]
            initial_samples = self.collect_data(idx, self.test_env, self.num_steps_posterior, 1, np.inf)
            #todo: there is a z gap between these sampling processes
            extra_samples = self.collect_data(idx, self.test_env, self.num_extra_rl_steps_posterior, 1, np.inf)
            self.test_buffer.add_traj(*initial_samples)
            self.test_buffer.add_traj(*extra_samples)
            context = self.sample_context([idx], train=False) 
            z_means, z_vars = self.agent.infer_z_posterior(context)
            z = self.agent.sample_z_from_posterior(z_means, z_vars)
            for traj_idx in range(self.num_test_trajectories):
                test_data = self.rollout_trajectory(self.test_env, z, deterministic=True)
                reward_list = test_data[3]
                traj_return = np.sum(reward_list)
                traj_returns.append(traj_return)
            test_traj_returns.append(np.mean(traj_returns))
        test_traj_mean_return = np.mean(test_traj_returns)
        return {"return/test": test_traj_mean_return
        }

    def collect_data(self, task_idx, env, num_samples, resample_z_rate, update_posterior_rate, z=None, sample_train_buffers=True):
        num_samples_collected = 0
        num_trajectories_collected = 0
        obs_list, action_list, next_obs_list, reward_list, done_list = [], [], [], [], []
        if z is None:
            z_means, z_vars = torch.zeros((1, self.agent.latent_dim)), torch.ones((1, self.agent.latent_dim))
        else:
            z_means, z_vars = z
        z_sample = self.agent.sample_z_from_posterior(z_means, z_vars)
        self.agent.set_z(z_sample)
        while num_samples_collected < num_samples:
            curr_obs_list, curr_action_list, curr_next_obs_list, curr_reward_list, curr_done_list = self.rollout_trajectory(env, z_sample, deterministic=True)
            obs_list += curr_obs_list
            action_list += curr_action_list
            next_obs_list += curr_next_obs_list
            reward_list += curr_reward_list
            done_list += curr_done_list
            num_samples_collected += len(curr_obs_list)
            num_trajectories_collected += 1
            if num_trajectories_collected % resample_z_rate == 0:
                z_sample = self.agent.sample_z_from_posterior(z_means, z_vars)
                self.agent.set_z(z_sample)
            if num_trajectories_collected % update_posterior_rate == 0:
                #update z posterior inference
                z_inference_context_batch = self.sample_context([task_idx], train=True)
                z_means, z_vars = self.agent.infer_z_posterior(z_inference_context_batch)
                z_sample = self.agent.sample_z_from_posterior(z_means, z_vars)
                self.agent.set_z(z_sample)
        return obs_list, action_list, next_obs_list, reward_list, done_list

    def rollout_trajectory(self, env, z, deterministic=False):
        obs_list, action_list, next_obs_list, reward_list, done_list = [], [], [], [], []
        done = False
        obs = env.reset()
        traj_length = 0
        while not done and traj_length < self.max_trajectory_length:
            action, log_prob = self.agent.select_action(obs, z, deterministic=deterministic)
            next_obs, reward, done, info = env.step(action)
            obs_list.append(obs)
            action_list.append(action)
            next_obs_list.append(next_obs)
            reward_list.append(reward)
            done_list.append(done)
            traj_length += 1
        return obs_list, action_list, next_obs_list, reward_list, done_list
        

    def save_video_demo(self, ite, width=128, height=128, fps=30):
        video_demo_dir = os.path.join(self.logger.log_dir,"demos")
        if not os.path.exists(video_demo_dir):
            os.makedirs(video_demo_dir)
        ite_video_demo_dir = os.path.join(video_demo_dir, "ite_{}".format(ite))
        os.mkdir(ite_video_demo_dir)
        video_size = (height, width)
        for idx in range(self.num_test_tasks):
            #reset env and buffer
            self.test_env.reset_task(idx)
            self.test_buffer.clear()
            initial_samples = self.collect_data(idx, self.test_env, self.num_steps_posterior, 1, np.inf)
            #todo: there is a z gap between these sampling processes
            extra_samples = self.collect_data(idx, self.test_env, self.num_extra_rl_steps_posterior, 1, np.inf)
            self.test_buffer.add_traj(*initial_samples)
            self.test_buffer.add_traj(*extra_samples)
            context = self.sample_context([1], train=False) 
            z_means, z_vars = self.agent.infer_z_posterior(context)
            z = self.agent.sample_z_from_posterior(z_means, z_vars)
            video_save_path = os.path.join(ite_video_demo_dir, "task_{}.avi".format(idx))

            #initilialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter(video_save_path, fourcc, fps, video_size)

            #rollout to generate pictures and write video
            state = self.test_env.reset()
            img = self.test_env.render(mode="rgb_array", width=width, height=height)
            for step in range(self.max_trajectory_length):
                action, _ = self.agent.select_action(state, z, deterministic=True)
                next_state, reward, done, _ = self.test_env.step(action)
                state = next_state
                img = self.test_env.render(mode="rgb_array", width=width, height=height)
                video_writer.write(img)
                if done:
                    break 
            video_writer.release()




            


