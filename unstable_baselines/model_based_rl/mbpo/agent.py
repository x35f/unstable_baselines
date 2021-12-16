import torch
import torch.nn.functional as F
import os
from unstable_baselines.common.agents import BaseAgent
from unstable_baselines.common.networks import MLPNetwork, PolicyNetworkFactory, get_optimizer
from unstable_baselines.common.models import EnsembleModel, EnvPredictor
import numpy as np
from unstable_baselines.common import util 
from operator import itemgetter

class MBPOAgent(torch.nn.Module, BaseAgent):
    def __init__(self,observation_space, action_space, env_name,
        target_smoothing_tau=0.1,
        alpha=0.2,
        **kwargs):
        obs_dim = observation_space.shape[0]
        action_dim = action_space.shape[0]
        super(MBPOAgent, self).__init__()
        #save parameters
        self.args = kwargs

        # get per flag
        self.per = self.args.get('per', False)

        #initilze networks
        self.q1_network = MLPNetwork(obs_dim + action_dim, 1, **kwargs['q_network'])
        self.q2_network = MLPNetwork(obs_dim + action_dim, 1,**kwargs['q_network'])
        self.target_q1_network = MLPNetwork(obs_dim + action_dim, 1,**kwargs['q_network'])
        self.target_q2_network = MLPNetwork(obs_dim + action_dim, 1,**kwargs['q_network'])
        self.policy_network = PolicyNetworkFactory.get(obs_dim, action_space,  ** kwargs['policy_network'])
        self.transition_model = EnsembleModel(obs_dim, action_dim, **kwargs['transition_model'])

        
        #sync network parameters
        util.soft_update_network(self.q1_network, self.target_q1_network, 1.0)
        util.soft_update_network(self.q2_network, self.target_q2_network, 1.0)

        #pass to util.device
        self.q1_network = self.q1_network.to(util.device)
        self.q2_network = self.q2_network.to(util.device)
        self.target_q1_network = self.target_q1_network.to(util.device)
        self.target_q2_network = self.target_q2_network.to(util.device)
        self.policy_network = self.policy_network.to(util.device)
        self.transition_model = self.transition_model.to(util.device)
        
        self.networks = {
            "q1": self.q1_network,
            "q2": self.q2_network,
            "target_q1": self.target_q1_network,
            "target_q2": self.target_q2_network,
            "policy": self.policy_network,
            "transition_model": self.transition_model
        }

        #initialize optimizer
        self.q1_optimizer = get_optimizer(network = self.q1_network, **kwargs['q_network'])
        self.q2_optimizer = get_optimizer(network = self.q2_network, **kwargs['q_network'])
        self.policy_optimizer = get_optimizer(network = self.policy_network, **kwargs['policy_network'])
        self.transition_optimizer = get_optimizer(network = self.transition_model, **kwargs['transition_model'])

        #initialize predict env
        self.env_predctor = EnvPredictor(self.transition_model, env_name=env_name)

        #hyper-parameters
        self.gamma = kwargs['gamma']
        self.automatic_entropy_tuning = kwargs['entropy']['automatic_tuning']
        self.alpha = alpha
        if self.automatic_entropy_tuning is True:
            self.target_entropy = -np.prod(action_space.shape).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=util.device)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=kwargs['entropy']['learning_rate'])
        self.tot_update_count = 0 
        self.target_smoothing_tau = target_smoothing_tau
        self.holdout_ratio = kwargs['transition_model']['holdout_ratio']
        self.inc_var_loss = kwargs['transition_model']['inc_var_loss']


    def train_model(self, data_batch):
        #compute the number of holdout samples
        batch_size = data_batch['obs'].shape[0]
        assert(batch_size == 100)
        num_holdout = int(batch_size * self.holdout_ratio)

        #permutate samples
        permutation = np.random.permutation(batch_size)
        obs_batch, action_batch, next_obs_batch, reward_batch, done_batch = \
            itemgetter("obs",'action','next_obs', 'reward', 'done')(data_batch)
        obs_batch, action_batch, next_obs_batch, reward_batch = \
            obs_batch[permutation], action_batch[permutation], next_obs_batch[permutation], reward_batch[permutation]

        #divide samples into training samples and testing samples
        train_obs_batch, train_action_batch, train_next_obs_batch, train_reward_batch = \
            obs_batch[num_holdout:], action_batch[num_holdout:], next_obs_batch[num_holdout:], reward_batch[num_holdout:]
        test_obs_batch, test_action_batch, test_next_obs_batch, test_reward_batch = \
            obs_batch[:num_holdout], action_batch[:num_holdout], next_obs_batch[:num_holdout], reward_batch[:num_holdout]

        #predict with model
        predictions = self.transition_model.predict(train_obs_batch, train_action_batch)
        
        #compute training loss
        obs_reward = torch.cat((train_next_obs_batch - train_obs_batch, train_reward_batch), dim=1)
        train_loss = sum(self.model_loss(predictions, obs_reward))
        
        #back propogate
        self.transition_optimizer.zero_grad()
        train_loss.backward()
        self.transition_optimizer.step()

        #compute testing loss
        with torch.no_grad():
            test_predictions = self.transition_model.predict(test_obs_batch, test_action_batch)
            test_obs_reward = torch.cat((test_next_obs_batch - test_obs_batch, test_reward_batch), dim=1)
            test_loss = sum(self.model_loss(test_predictions, test_obs_reward))
            idx = np.argsort(test_loss.detach().cpu())
            self.transition_model.elite_model_idxes = idx[:self.transition_model.num_elite]
        
        return {
            "train_transition_loss": train_loss,
            "test_transition_loss": test_loss
        }

    def model_loss(self, predictions, trues):
        loss = []
        for (means, vars) in predictions:
            if self.inc_var_loss:
                mean_loss = torch.mean(torch.mean(torch.pow(means - trues, 2) * (1/vars), dim=-1), dim=-1)
                var_loss = torch.mean(torch.mean(torch.log(vars), dim=-1), dim=-1)
                loss.append(mean_loss + var_loss)
            else:
                loss.append(torch.mean(torch.mean(torch.pow(means - trues, 2), dim=-1), dim=-1))
        return loss


    def update(self, data_batch):
        obs_batch = data_batch['obs']
        action_batch = data_batch['action'] 
        next_obs_batch = data_batch['next_obs'] 
        reward_batch = data_batch['reward']
        done_batch = data_batch['done']
        
        curr_state_q1_value = self.q1_network(torch.cat([obs_batch, action_batch],dim=1))
        curr_state_q2_value = self.q2_network(torch.cat([obs_batch, action_batch],dim=1))
        next_state_action, next_state_log_pi = \
            itemgetter("action_scaled", "log_prob")(self.policy_network.sample(next_obs_batch))

        next_state_q1_value = self.target_q1_network(torch.cat([next_obs_batch, next_state_action], dim=1))
        next_state_q2_value = self.target_q2_network(torch.cat([next_obs_batch, next_state_action], dim=1))
        next_state_min_q = torch.min(next_state_q1_value, next_state_q2_value)
        target_q = (next_state_min_q - self.alpha * next_state_log_pi)
        target_q = reward_batch + self.gamma * (1. - done_batch) * target_q

        #compute q loss and backward
        
        q1_loss = F.mse_loss(curr_state_q1_value, target_q.detach())
        q2_loss = F.mse_loss(curr_state_q2_value, target_q.detach())

        q1_loss_value = q1_loss.detach().cpu().numpy()
        q2_loss_value = q2_loss.detach().cpu().numpy()

        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        (q1_loss + q2_loss).backward()
        self.q1_optimizer.step()
        self.q2_optimizer.step()

        ##########

        new_curr_state_action, new_curr_state_log_pi = \
            itemgetter("action_scaled", "log_prob")(self.policy_network.sample(obs_batch))
        new_curr_state_q1_value = self.q1_network(torch.cat([obs_batch, new_curr_state_action],dim=1))
        new_curr_state_q2_value = self.q2_network(torch.cat([obs_batch, new_curr_state_action],dim=1))
        new_min_curr_state_q_value = torch.min(new_curr_state_q1_value, new_curr_state_q2_value)
        
        #compute policy and ent loss
        policy_loss = ((self.alpha * new_curr_state_log_pi) - new_min_curr_state_q_value).mean()
        policy_loss_value = policy_loss.detach().cpu().numpy()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (new_curr_state_log_pi + self.target_entropy).detach()).mean()
            alpha_loss_value = alpha_loss.detach().cpu().item()
            self.alpha_optim.zero_grad()
        else:
            alpha_loss = 0.
            alpha_loss_value = 0.
        
        self.policy_optimizer.zero_grad()
        (policy_loss + alpha_loss).backward()
        self.policy_optimizer.step()
        if self.automatic_entropy_tuning:
            self.alpha_optim.step()
            self.alpha = self.log_alpha.detach().exp()
            alpha_value = self.alpha.cpu().numpy()
        else:
            alpha_value = self.alpha

        # backward and step
        self.tot_update_count += 1

        self.try_update_target_network()
        
        return {
            "loss/q1": q1_loss_value, 
            "loss/q2": q2_loss_value, 
            "loss/policy": policy_loss_value, 
            "loss/entropy": alpha_loss_value, 
            "others/entropy_alpha": alpha_value, 
            "misc/current_state_q1_value": torch.norm(curr_state_q1_value.squeeze().detach().clone().cpu(), p=1) / len(curr_state_q1_value.squeeze()), 
            "misc/current_state_q2_value": torch.norm(curr_state_q2_value.squeeze().detach().clone().cpu(), p=1) / len(curr_state_q2_value.squeeze()),
            "misc/q_diff": torch.norm((curr_state_q2_value-curr_state_q1_value).squeeze().detach().clone().cpu(), p=1) / len(curr_state_q1_value.squeeze())
        }
        

    def try_update_target_network(self):
        util.soft_update_network(self.q1_network, self.target_q1_network, self.target_smoothing_tau)
        util.soft_update_network(self.q2_network, self.target_q2_network, self.target_smoothing_tau)

    @torch.no_grad()  
    def select_action(self, obs, deterministic=False):
        if len(obs.shape) == 1:
            obs = obs[None, :]
    
        if not isinstance(obs, torch.Tensor):
            obs = torch.FloatTensor(obs).to(util.device)

        action_scaled, log_prob = \
                itemgetter("action_scaled", "log_prob")(self.policy_network.sample(obs, deterministic))
        if len(obs.shape) == 1:
            action_scaled = action_scaled.squeeze()
        return action_scaled.detach().cpu().numpy(), log_prob.detach().cpu().numpy()


    def rollout(self, obs_batch, model_rollout_steps):
        obs_list = np.array([])
        action_list = np.array([])
        next_obs_list = np.array([])
        reward_list = np.array([])
        done_list = np.array([])
        obs = obs_batch.detach().cpu().numpy()
        for step in range(model_rollout_steps):
            action, log_prob = self.select_action(obs)
            next_obs, reward, done, info = self.env_predctor.predict(obs, action)
            if step == 0:
                obs_list = obs
                action_list = action
                next_obs_list = next_obs
                reward_list = reward
                done_list = done
            else:
                obs_list = np.concatenate((obs_list, obs), 0)
                action_list = np.concatenate((action_list, action), 0)
                next_obs_list = np.concatenate((next_obs_list, next_obs), 0)
                reward_list = np.concatenate((reward_list, reward), 0)
                done_list = np.concatenate((done_list, done), 0)
            obs = np.array([obs_pred for obs_pred, done_pred in zip(next_obs_list, done_list) if not done_pred[0]])
        #assert(obs_list.shape[1] == 11)
        return {'obs_list': obs_list, 'action_list': action_list, 'reward_list': reward_list, 'next_obs_list': next_obs_list, 'done_list': done_list}

        



