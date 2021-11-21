import torch
import torch.nn.functional as F
import gym 
import os
from torch import mean, nn, var
from common.agents import BaseAgent
from common.networks import MLPNetwork, PolicyNetwork, get_optimizer
from common.buffer import ReplayBuffer
from common.models import BaseModel, EnsembleModel, PredictEnv
import numpy as np
from common import util 

class MBPOAgent(torch.nn.Module, BaseAgent):
    def __init__(self,observation_space, action_space,
        update_target_network_interval=50, 
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
        self.policy_network = PolicyNetwork(obs_dim, action_space,  ** kwargs['policy_network'])
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

        #initialize optimizer
        self.q1_optimizer = get_optimizer(network = self.q1_network, **kwargs['q_network'])
        self.q2_optimizer = get_optimizer(network = self.q2_network, **kwargs['q_network'])
        self.policy_optimizer = get_optimizer(network = self.policy_network, **kwargs['policy_network'])
        self.transition_optimizer = get_optimizer(network = self.transition_model, **kwargs['transition_model'])

        #initialize predict env
        self.predict_env = PredictEnv(self.transition_model, kwargs['transition_model']['env_name'])

        #hyper-parameters
        self.gamma = kwargs['gamma']
        self.automatic_entropy_tuning = kwargs['entropy']['automatic_tuning']
        self.alpha = alpha
        if self.automatic_entropy_tuning is True:
            self.target_entropy = -np.prod(action_space.shape).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=util.device)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=kwargs['entropy']['learning_rate'])
        self.tot_update_count = 0 
        self.update_target_network_interval = update_target_network_interval
        self.target_smoothing_tau = target_smoothing_tau
        self.holdout_ratio = kwargs['transition_model']['holdout_ratio']
        self.inc_var_loss = kwargs['transition_model']['inc_var_loss']


    def train_model(self, data_batch):
        #compute the number of holdout samples
        batch_size = data_batch[0].shape[0]
        assert(batch_size == 100)
        num_holdout = int(batch_size * self.holdout_ratio)

        #permutate samples
        permutation = np.random.permutation(batch_size)
        state_batch, action_batch, next_state_batch, reward_batch, done_batch = data_batch
        state_batch, action_batch, next_state_batch, reward_batch = \
            state_batch[permutation], action_batch[permutation], next_state_batch[permutation], reward_batch[permutation]

        #divide samples into training samples and testing samples
        train_state_batch, train_action_batch, train_next_state_batch, train_reward_batch = \
            state_batch[num_holdout:], action_batch[num_holdout:], next_state_batch[num_holdout:], reward_batch[num_holdout:]
        test_state_batch, test_action_batch, test_next_state_batch, test_reward_batch = \
            state_batch[:num_holdout], action_batch[:num_holdout], next_state_batch[:num_holdout], reward_batch[:num_holdout]

        #predict with model
        predictions = self.transition_model.predict(train_state_batch, train_action_batch)
        
        #compute training loss
        state_reward = torch.cat((train_next_state_batch - train_state_batch, train_reward_batch), dim=1)
        train_loss = sum(self.model_loss(predictions, state_reward))
        
        #back propogate
        self.transition_optimizer.zero_grad()
        train_loss.backward()
        self.transition_optimizer.step()

        #compute testing loss
        with torch.no_grad():
            test_predictions = self.transition_model.predict(test_state_batch, test_action_batch)
            test_state_reward = torch.cat((test_next_state_batch - test_state_batch, test_reward_batch), dim=1)
            test_loss = sum(self.model_loss(test_predictions, test_state_reward))
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
        if self.per:
            state_batch, action_batch, next_state_batch, reward_batch, done_batch, IS_batch, info_batch = data_batch
        else:
            state_batch, action_batch, next_state_batch, reward_batch, done_batch = data_batch
        
        curr_state_q1_value = self.q1_network(state_batch, action_batch)
        curr_state_q2_value = self.q2_network(state_batch, action_batch)
        new_curr_state_action, new_curr_state_log_pi, _ = self.policy_network.sample(state_batch)
        next_state_action, next_state_log_pi, _ = self.policy_network.sample(next_state_batch)

        new_curr_state_q1_value = self.q1_network(state_batch, new_curr_state_action)
        new_curr_state_q2_value = self.q2_network(state_batch, new_curr_state_action)

        next_state_q1_value = self.target_q1_network(next_state_batch, next_state_action)
        next_state_q2_value = self.target_q2_network(next_state_batch, next_state_action)
        next_state_min_q = torch.min(next_state_q1_value, next_state_q2_value)
        target_q = (next_state_min_q - self.alpha * next_state_log_pi)
        target_q = reward_batch + self.gamma * (1. - done_batch) * target_q

        new_min_curr_state_q_value = torch.min(new_curr_state_q1_value, new_curr_state_q2_value)

        #compute q loss
        if self.per:
            # maybe average is better？
            q1_loss = torch.mean(IS_batch*torch.square(curr_state_q1_value-target_q.detach()))
            q2_loss = torch.mean(IS_batch*torch.square(curr_state_q2_value-target_q.detach()))
            abs_errors = torch.abs(target_q - 0.5*curr_state_q1_value - 0.5*curr_state_q2_value).detach().cpu().numpy().squeeze()
        else:
            q1_loss = F.mse_loss(curr_state_q1_value, target_q.detach())
            q2_loss = F.mse_loss(curr_state_q2_value, target_q.detach())

        q1_loss_value = q1_loss.detach().cpu().numpy()
        q2_loss_value = q2_loss.detach().cpu().numpy()
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q2_optimizer.zero_grad()
        q2_loss.backward()

        #compute policy loss
        policy_loss = ((self.alpha * new_curr_state_log_pi) - new_min_curr_state_q_value).mean()
        policy_loss_value = policy_loss.detach().cpu().numpy()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()

        #compute entropy loss
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (new_curr_state_log_pi + self.target_entropy).detach()).mean()
            alpha_loss_value = alpha_loss.detach().cpu().numpy()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_value = self.alpha.detach().cpu().numpy()
        else:
            alpha_loss_value = 0.
            alpha_value = self.alpha
        self.tot_update_count += 1

        self.q1_optimizer.step()
        self.q2_optimizer.step()
        self.policy_optimizer.step()
        
        if self.per:
            #　need to return new abs TD errors to update samples in buffer
            return {
                "loss/q1": q1_loss_value, 
                "loss/q2": q2_loss_value, 
                "loss/policy": policy_loss_value, 
                "loss/entropy": alpha_loss_value, 
                "others/entropy_alpha": alpha_value, 
            }, abs_errors
        else:
            return {
                "loss/q1": q1_loss_value, 
                "loss/q2": q2_loss_value, 
                "loss/policy": policy_loss_value, 
                "loss/entropy": alpha_loss_value, 
                "others/entropy_alpha": alpha_value
            }
        

    def try_update_target_network(self):
        if self.tot_update_count % self.update_target_network_interval == 0:
            util.soft_update_network(self.q1_network, self.target_q1_network, self.target_smoothing_tau)
            util.soft_update_network(self.q2_network, self.target_q2_network, self.target_smoothing_tau)
            
    def select_action(self, state, evaluate=False, step=1):
        if type(state) != torch.Tensor:
            if len(state.shape) == 1:
                state = torch.FloatTensor([state]).to(util.device)
            else:
                state = torch.FloatTensor(state).to(util.device)
        action, log_prob, mean = self.policy_network.sample(state)
        if evaluate:
            return mean.detach().cpu().numpy(), log_prob
        else:
            return action.detach().cpu().numpy(), log_prob


    def save_model(self, target_dir, ite):
        target_dir = os.path.join(target_dir, "ite_{}".format(ite))
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        #save q networks 
        save_path = os.path.join(target_dir, "Q_network_1.pt")
        torch.save(self.q1_network, save_path)
        save_path = os.path.join(target_dir, "Q_network_2.pt")
        torch.save(self.q2_network, save_path)
        #save policy network
        save_path = os.path.join(target_dir, "policy_network.pt")
        torch.save(self.policy_network, save_path)


    def load_model(self, model_dir):
        q1_network_path = os.path.join(model_dir, "Q_network_1.pt")
        self.q1_network.load_state_dict(torch.load(q1_network_path))
        q2_network_path = os.path.join(model_dir, "Q_network_2.pt")
        self.q2_network.load_state_dict(torch.load(q2_network_path))
        policy_network_path = os.path.join(model_dir, "policy_network.pt")
        self.policy_network.load_state_dict(torch.load(policy_network_path))

    def rollout(self, state_batch, model_rollout_steps):
        state_set = np.array([])
        action_set = np.array([])
        next_state_set = np.array([])
        reward_set = np.array([])
        done_set = np.array([])
        state = state_batch.detach().cpu().numpy()
        for step in range(model_rollout_steps):
            action, log_prob = self.select_action(state)
            next_state, reward, done, info = self.predict_env.step(state, action)
            if step == 0:
                state_set = state
                action_set = action
                next_state_set = next_state
                reward_set = reward
                done_set = done
            else:
                state_set = np.concatenate((state_set, state), 0)
                action_set = np.concatenate((action_set, action), 0)
                next_state_set = np.concatenate((next_state_set, next_state), 0)
                reward_set = np.concatenate((reward_set, reward), 0)
                done_set = np.concatenate((done_set, done), 0)
        assert(state_set.shape[1] == 11)
        return {'state': state_set, 'action': action_set, 'reward': reward_set, 'next_state': next_state_set, 'done': done_set}

        



