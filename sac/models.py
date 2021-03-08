import torch
from util import REPLAY_BUFFER, soft_update_network
import gym 
from torch import nn
import util
import random
import numpy as np
class SACAgent(torch.nn.Module):
    def __init__(self, logger,env,args
    ):
        super().__init__()
        assert(type(env.action_space) == gym.spaces.discrete.Discrete)
        #save parameters
        self.logger = logger
        self.env = env
        self.args = args
        #initilize replay buffer
        self.replay_buffer = REPLAY_BUFFER(self.obs_dim, self.action_dim, max_buffer_size)
        #initilze networks
        self.Q_target_network = MLP(self.obs_dim, self.action_dim, output_fn = nn.Identity, depth = network_depth, hidden_size = network_width, act_fn = nn.Tanh)
        self.Q_network = MLP(self.obs_dim, self.action_dim, output_fn = nn.Identity, depth = network_depth, hidden_size = network_width, act_fn = nn.Tanh)
        #initialize optimizer
        
        #pass to util.device
        self.Q_target_network = self.Q_target_network.to(util.device)
        self.Q_network = self.Q_network.to(util.device)
    def init_value_network()
    def train(self):
        tot_num_update = 0
        for epoch in range(self.max_epoch):
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

    def test(self,ite):
        traj_rewards = []
        for traj_id in range(self.num_test_trajs):
            obs = self.env.reset()
            traj_reward = 0
            #rollout in environment
            for step in range(self.max_traj_length):
                action = self.select_action(obs)
                next_obs, reward, done, info = self.env.step(action)
                traj_reward += reward
                obs = next_obs
                if done:
                    break
            traj_rewards.append(traj_reward)
        mean_reward = np.mean(traj_rewards)
        #print(traj_rewards)
        self.logger.log_var("performance/test_return",mean_reward, ite )
        self.logger.log_str("Iteration {}\t:\tTest average return {:02f}\t{}".format(ite, mean_reward,traj_rewards))
            
    def select_action(self, obs):
        ob = torch.tensor(obs).to(util.device).unsqueeze(0).float()
        Q_values = self.Q_network(ob)
        Q_values, action_indices = torch.max(Q_values, dim=1)
        return action_indices.detach().cpu().numpy()[0]



class MLP(nn.Module):
    def __init__(self, 
            input_dim, 
            output_dim,
            hidden_size = 128,
            depth = 1,
            act_fn = torch.nn.ReLU,
            output_fn = torch.nn.Sigmoid
        ):
        super().__init__()
        self.act_fn = act_fn()
        self.fc_init = nn.Linear(input_dim, hidden_size)
        self.fcs = [nn.Linear(hidden_size, hidden_size) for _ in range(depth)]
        self.fc_final = nn.Linear(hidden_size, output_dim)
        self.output_fn = output_fn()

    def forward(self,input_data):
        x = self.fc_init(input_data)
        x = self.act_fn(x)
        for fc in self.fcs:
            x = fc(x)
            x = self.act_fn(x)
        x = self.fc_final(x)
        x = self.output_fn(x)
        return x
