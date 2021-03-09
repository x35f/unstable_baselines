from common.util import hard_update_network, soft_update_network
from torch.nn.functional import max_pool1d_with_indices
from common.trainer import BaseTrainer
import numpy as np
from tqdm import tqdm
from time import time
class SACTrainer(BaseTrainer):
    def __init__(self, agent, env, buffer, logger, **kwargs):
        self.agent = agent
        self.buffer = buffer
        self.logger = logger
        self.env = env 
        #hyperparameters
        self.batch_size = kwargs['batch_size']
        self.num_updates_per_ite = kwargs['num_updates_per_iteration']
        self.max_traj_length = kwargs['max_trajectory_length']
        self.test_interval = kwargs['test_interval']
        self.trajs_per_test = kwargs['num_test_trajectories']
        self.update_v_target_interval = kwargs['update_v_target_interval']
        self.target_smoothing_tau = kwargs['target_smoothing_tau']
        self.max_episode = kwargs['max_episode']


    def train(self):
        tot_num_updates = 0
        train_traj_rewards = []
        episode_durations = []

        for episode in range(self.max_episode):
            episode_start_time = time()
            #rollout in environment and add to buffer
            state = self.env.reset()
            done = False
            traj_reward = 0
            traj_length = 0
            for i in tqdm(range(self.max_traj_length)):
                action = self.agent.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                if i == self.max_traj_length - 1:
                    done = True
                traj_length  += 1
                traj_reward += reward
                self.buffer.add_tuple(state, action, next_state, reward, float(done))
                state = next_state
                if done:
                    break
            train_traj_rewards.append(traj_reward)
            self.logger.log_var("return/train",traj_reward,tot_num_updates)
            #update network
            for update in range(self.num_updates_per_ite):
                data_batch = self.buffer.sample_batch(self.batch_size)
                q_loss1, q_loss2, v_loss, policy_loss, entropy_loss, alpha = self.agent.update(data_batch)
                self.logger.log_var("loss/q1",q_loss1,tot_num_updates)
                self.logger.log_var("loss/q2",q_loss2,tot_num_updates)
                self.logger.log_var("loss/v",v_loss,tot_num_updates)
                self.logger.log_var("loss/policy",policy_loss,tot_num_updates)
                self.logger.log_var("loss/entropy",entropy_loss,tot_num_updates)
                self.logger.log_var("others/entropy_alpha",alpha,tot_num_updates)
                tot_num_updates += 1

            if tot_num_updates % self.test_interval == 0:
                avg_test_reward = self.test()
                self.logger.log_var("return/test", avg_test_reward, tot_num_updates)

            if episode % self.update_v_target_interval == 0:
                soft_update_network(self.agent.v_network, self.agent.target_v_network, self.target_smoothing_tau)

            episode_end_time = time()
            episode_duration = episode_end_time - episode_start_time
            episode_durations.append(episode_duration)
            print("episode {}: return {} eta: {}s".format(episode, np.mean(train_traj_rewards[-2:], int(np.mean(episode_durations[-5:])))))



    def test(self):
        rewards = []
        for i in range(self.trajs_per_test):
            traj_reward = 0
            state = self.env.reset()
            for i in range(self.max_traj_length):
                action = self.agent.select_action(state, evaluate=True)
                next_state, reward, done, _ = self.env.step(action)
                traj_reward += reward
                state = next_state
                if done:
                    break
            rewards.append(traj_reward)
        return np.mean(rewards)
            


