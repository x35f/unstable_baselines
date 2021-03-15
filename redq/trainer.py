from common.util import hard_update_network, soft_update_network, second_to_time_str
from torch.nn.functional import max_pool1d_with_indices
from common.trainer import BaseTrainer
import numpy as np
from tqdm import tqdm
from time import time
class SACTrainer(BaseTrainer):
    def __init__(self, agent, env, buffer, logger, 
            batch_size=32,
            num_updates_per_iteration=20,
            max_trajectory_length=500,
            test_interval=10,
            num_test_trajectories=5,
            max_episode=100000,
            **kwargs):
        self.agent = agent
        self.buffer = buffer
        self.logger = logger
        self.env = env 
        #hyperparameters
        self.batch_size = batch_size
        self.num_updates_per_ite = num_updates_per_iteration
        self.max_trajectory_length = max_trajectory_length
        self.test_interval = test_interval
        self.num_test_trajectories = num_test_trajectories
        self.max_episode = max_episode


    def train(self):
        tot_num_updates = 0
        train_traj_rewards = []
        episode_durations = []

        for episode in range(self.max_episode):
            episode_start_time = time()
            #rollout in environment and add to buffer
            state = self.env.reset()
            traj_reward = 0
            traj_length = 0
            for step in range(self.max_trajectory_length):
                action = self.agent.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                if step == self.max_trajectory_length - 1:
                    done = True
                traj_length  += 1
                traj_reward += reward
                self.buffer.add_tuple(state, action, next_state, reward, float(done))
                state = next_state
                if done:
                    break
            traj_reward /= self.env.reward_scale
            train_traj_rewards.append(traj_reward)
            self.logger.log_var("return/train",traj_reward, tot_num_updates)
            #update network
            for update in range(self.num_updates_per_ite):
                data_batch = self.buffer.sample_batch(self.batch_size)
                q_loss1, q_loss2, policy_loss, entropy_loss, alpha = self.agent.update(data_batch)
                self.logger.log_var("loss/q1",q_loss1,tot_num_updates)
                self.logger.log_var("loss/q2",q_loss2,tot_num_updates)
                self.logger.log_var("loss/policy",policy_loss,tot_num_updates)
                self.logger.log_var("loss/entropy",entropy_loss,tot_num_updates)
                self.logger.log_var("others/entropy_alpha",alpha,tot_num_updates)
                self.agent.try_update_target_network()
                tot_num_updates += 1
       
            episode_end_time = time()
            episode_duration = episode_end_time - episode_start_time
            episode_durations.append(episode_duration)

            if tot_num_updates % self.test_interval == 0:
                avg_test_reward, avg_test_length = self.test()
                self.logger.log_var("return/test", avg_test_reward, tot_num_updates)
                self.logger.log_var("return/test_length", avg_test_length, tot_num_updates)
                time_remaining_str = second_to_time_str(int((self.max_episode - episode + 1) * np.mean(episode_durations[-100:])))
                summary_str = "episode {}:\ttrain return {:.02f}\ttest return {:02f}\teta: {}".format(episode, train_traj_rewards[-1],avg_test_reward,time_remaining_str)
                self.logger.log_str(summary_str)


    def test(self):
        rewards = []
        lengths = []
        for episode in range(self.num_test_trajectories):
            traj_reward = 0
            traj_length = 0
            state = self.env.reset()
            for step in range(self.max_trajectory_length):
                action = self.agent.select_action(state, evaluate=True)
                next_state, reward, done, _ = self.env.step(action)
                traj_reward += reward
                state = next_state
                traj_length += 1 
                if done:
                    break
            lengths.append(traj_length)
            traj_reward /= self.env.reward_scale
            rewards.append(traj_reward)
        return np.mean(rewards), np.mean(lengths)
            


