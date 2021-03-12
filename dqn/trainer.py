from common.util import hard_update_network, soft_update_network, second_to_time_str
from torch.nn.functional import max_pool1d_with_indices
from common.trainer import BaseTrainer
import numpy as np
from tqdm import tqdm
from time import time
class DQNTrainer(BaseTrainer):
    def __init__(self, agent, env, buffer, logger, 
            batch_size=32,
            num_updates_per_iteration=20,
            max_trajectory_length=500,
            test_interval=10,
            num_test_trajectories=5,
            update_v_target_interval=5,
            target_smoothing_tau=0.6,
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
        self.update_v_target_interval = update_v_target_interval
        self.target_smoothing_tau = target_smoothing_tau
        self.max_episode = max_episode


    def train(self):
       tot_num_update = 0
        for epoch in range(self.max_episode):
            obs = self.env.reset()
            #rollout in environment
            traj_rewards = []
            curr_traj_reward = 0
            tot_sample_steps = 0
            while tot_sample_steps < self.num_sample_steps_per_epoch:
                obs = self.env.reset()
                curr_traj_reward = 0
                for step in range(self.max_traj_length):
                    if random.random() < self.epsilon:
                        action = self.env.action_space.sample()
                    else: 
                        action = self.select_action(obs)
                    next_obs, reward, done, info = self.env.step(action)
                    self.replay_buffer.add_tuple(obs, action, next_obs, reward, done)
                    obs = next_obs
                    curr_traj_reward += reward
                    tot_sample_steps += 1
                    if done or tot_sample_steps > self.num_sample_steps_per_epoch or step >= self.max_traj_length-1:
                        traj_rewards.append(curr_traj_reward)
                        break
            mean_reward = np.mean(traj_rewards)
            self.logger.log_var("performance/train_return", np.mean(mean_reward), tot_num_update)
            for update_step in range(self.num_update_steps_per_epoch):
                obs_batch, action_batch, next_obs_batch, reward_batch, done_batch = self.replay_buffer.sample_batch(self.batch_size, to_tensor = True)
                
                #compute Q_target
                with torch.no_grad():
                    Q_target_values = self.Q_target_network(next_obs_batch)
                    Q_target_values, Q_target_actions = torch.max(Q_target_values, dim =1)
                    Q_target = reward_batch + (1 - done_batch) * self.gamma * Q_target_values
                
                #compute Q current
                Q_current_values = self.Q_network(obs_batch)
                Q_current = torch.stack([_[idx] for _, idx in zip(Q_current_values, action_batch)])
                
                #compute loss
                loss = self.loss_fn(Q_target, Q_current)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                tot_num_update += 1
                loss_val = loss.detach().cpu().numpy()
                self.logger.log_var("losses/train_loss",loss_val, tot_num_update)
                if tot_num_update % self.update_q_target_inverval:
                    soft_update_network(self.Q_network, self.Q_target_network, self.tau)

            if epoch % self.test_interval == 0:
                self.test(tot_num_update)
                #self.logger.log_str("Epoch:\t{}\tLoss:\t{}\tReturn:\t{}".format(epoch,loss_val,mean_reward))


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
            


