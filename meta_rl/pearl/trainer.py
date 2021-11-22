from common.util import second_to_time_str
from common.trainer import BaseTrainer
import numpy as np
from tqdm import tqdm
from time import time
import cv2
import os
from tqdm import  tqdm

class SACTrainer(BaseTrainer):
    def __init__(self, agent, env, eval_env, buffer, logger, load_dir,
            batch_size=32,
            max_iteration=500,
            num_tasks_per_gradient_update=16,
            num_train_tasks=50,
            num_test_tasks=10,
            num_tasks_per_iteration=5, 
            num_steps_prior=400, 
            num_steps_posterior=0, 
            num_extra_rl_steps_posterior=400,
            num_updates_per_iteration=2000,
            num_evaluations=2,
            num_steps_per_evaluation=600,
            adaptation_context_update_interval=1,
            num_adaptation_trajectories_before_posterior_sampling=1,
            num_steps_per_iteration=50,
            max_trajectory_length=200,
            test_interval=20,
            num_test_trajectories=5,
            save_model_interval=2000,
            start_timestep=2000,
            save_video_demo_interval=10000,
            log_interval=50,
            **kwargs):
        print("redundant arguments for trainer: {}".format(kwargs))
        self.agent = agent
        self.buffer = buffer
        self.logger = logger
        self.env = env 
        self.eval_env = eval_env
        #hyperparameters
        self.batch_size = batch_size
        self.num_train_tasks = num_train_tasks
        self.num_test_tasks = num_test_tasks
        self.max_iteration = max_iteration
        self.num_tasks_per_gradient_update = num_tasks_per_gradient_update
        self.num_tasks_per_iteration = num_tasks_per_iteration, 
        self.num_steps_prior = num_steps_prior
        self.num_steps_posterior = num_steps_posterior
        self.num_extra_rl_steps_posterior = num_extra_rl_steps_posterior
        self.num_updates_per_iteration = num_updates_per_iteration
        self.num_evaluations = num_evaluations
        self.num_steps_per_evaluation = num_steps_per_evaluation,
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
        train_traj_lengths = [0]
        iteration_durations = []
        tot_env_steps = 0
        state = self.env.reset()
        traj_reward = 0
        traj_length = 0
        done = False
        state = self.env.reset()
        for ite in tqdm(range(self.max_iteration)): 
            iteration_start_time = time()
            
            if ite == 0: # collect initial pool of data
                for idx in self.train_tasks:
                    self.env.reset_task(idx)
                    self.agent.clear_z()
                    self.collect_data(idx, self.start_timestep, 1, np.inf)

            #sample data from train_tasks
            for train_idx in range(self.num_train_tasks):
                train_task_idx = np.random.randint(self.num_train_tasks)
                self.env.reset_task(train_task_idx)
                self.enc_replay_buffer.task_buffers[train_task_idx].clear()

                # collect some trajectories with z ~ prior
                if self.num_steps_prior > 0:
                    self.agent.clear_z()
                    prior_data = self.collect_data(self.num_steps_prior, 1, np.inf)
                    #todo: add data to buffer

                # collect some trajectories with z ~ posterior
                if self.num_steps_posterior > 0:
                    self.agent.clear_z()
                    self.collect_data(self.num_steps_posterior, 1, self.adaptation_context_update_interval)
                    #todo: add data to buffer

                # the policy needs to learn to handle z ~ posterior
                if self.num_extra_rl_steps_posterior > 0:
                    self.agent.clear_z()
                    self.collect_data(self.num_extra_rl_steps_posterior, 1, self.adaptation_context_update_interval)
                    #todo: add data to buffer       

            #perform training on batches of train tasks
            for train_step in range(self.num_updates_per_iteration):
                train_task_indices = np.random.choice(self.train_tasks, self.num_tasks_per_gradient_update)
                loss_dict = self.train_step(train_task_indices)

            #post iteration logging
            iteration_end_time = time()
            iteration_duration = iteration_end_time - iteration_start_time
            iteration_durations.append(iteration_duration)
            if ite % self.log_interval == 0:
                for loss_name in loss_dict:
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

    def train_step(self, train_task_indices):
        # sample train context
        context_batch = self.sample_context(train_task_indices)
        
        #sample data for sac update
        data_batch = [self.train_buffers[index].sample(batch_size=self.batch_size) for index in train_task_indices]

        #clear z inference
        self.agent.clear_z(num_tasks = len(train_task_indices))
        self.agent.train(context_batch, data_batch)
        self.agent.detach_z()


    def sample_context(self):
        pass

    def collect_data(self, num_samples, resample_z_rate, update_posterior_rate):
        num_samples_collected = 0
        num_trajectories_collected = 0
        samples = []
        while num_samples_collected < num_samples:
            trajectory = self.rollout_single_trajectory()
            samples += trajectory
            num_samples_collected += len(trajectory)
            num_trajectories_collected += 1
            if num_trajectories_collected % resample_z_rate == 0:
                #resample z 

            if num_trajectories_collected % update_posterior_rate == 0:
                #update z posterior inference
                context = 
        pass

    def rollout_single_trajectory(self):
        pass

    def sample_context(self, indices):
        if type(indices) == int:
            indices = [indices]
        # sample context transitions from encoder buffer
        context_batches = [self.encoder_buffer[index].sample(batch) for index in indices]
        pass
        

    def test(self):
        pass

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




            


