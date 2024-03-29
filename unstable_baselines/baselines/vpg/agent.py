from operator import itemgetter

import torch
import gym
from operator import itemgetter
import numpy as np
from unstable_baselines.common import util
from unstable_baselines.common.agents import BaseAgent
from unstable_baselines.common.networks import SequentialNetwork
from unstable_baselines.common.networks import PolicyNetworkFactory
from unstable_baselines.common.networks import get_optimizer


class VPGAgent(BaseAgent):
    """ Vanilla Policy Gradient Agent
    https://proceedings.neurips.cc/paper/2001/file/4b86abe48d358ecf194c56c69108433e-Paper.pdf

    BaseAgent Args
    --------------
    observation_space: gym.Space

    action_sapce: gym.Space

    kwargs Args
    -----------
    gamma: float
        Discount factor.

    train_v_iters: int
        The number of times that the state-value network is updated in the agent.update 
        function, while the policy network is only updated once.

    action_bound_method: str, optional("clip", "tanh"),
        Method for mappolicyng the raw action generated by policy network to the environment 
        action space.
    """

    def __init__(self,
                observation_space: gym.Space,
                action_space: gym.Space,
                train_v_iters: int,
                gamma: float,
                action_bound_method: str,
                advantage_type: str,
                advantage_params: float,
                normalize_advantage: bool,
                 **kwargs):
        
        super(VPGAgent, self).__init__()
        # save parameters
        self.args = kwargs
        
        self.observation_space = observation_space
        self.action_space = action_space
        obs_shape = observation_space.shape
        if isinstance(action_space, gym.spaces.discrete.Discrete):
            self.discrete_action_space = True
        else:
            self.discrete_action_space = False

        # initialize networks and optimizer
        self.v_network = SequentialNetwork(obs_shape, 1, **kwargs['v_network']).to(util.device)
        self.v_optimizer = get_optimizer(kwargs['v_network']['optimizer_class'], self.v_network, kwargs['v_network']['learning_rate'])
        self.policy_network = PolicyNetworkFactory.get(observation_space, action_space, **kwargs['policy_network']).to(util.device)
        self.policy_optimizer = get_optimizer(kwargs['policy_network']['optimizer_class'], self.policy_network, kwargs['policy_network']['learning_rate'])
        
        # hyper-parameters
        self.gamma = gamma
        self.train_v_iters = train_v_iters
        self.action_bound_method = action_bound_method

        #advantage related parameters
        self.advantage_type = advantage_type
        self.normalize_advantage = normalize_advantage
        self.advantage_params = advantage_params

    

    def estimate_value(self, obs):
        """ Estimate the obs value.
        """
        if len(obs.shape) in [1,3]:
            obs = obs[None,]
    
        if not isinstance(obs, torch.Tensor):
            obs = torch.FloatTensor(obs).to(util.device)
        with torch.no_grad():
            value = self.v_network(obs)
        return value.detach().cpu().numpy()
    
    @torch.no_grad()
    def estimate_advantage(self, obs_batch, action_batch, reward_batch, next_obs_batch, done_batch, truncated_batch):
        if self.advantage_type == "gae":
            gae_lambda = self.advantage_params['lambda']
            value_batch = self.v_network(obs_batch)
            next_value_batch = self.v_network(next_obs_batch)
            advantage_batch = torch.zeros(action_batch.shape[0], 1).to(util.device)
            return_batch = torch.zeros(action_batch.shape[0], 1).to(util.device)
           
            delta_batch = reward_batch + next_value_batch * self.gamma - value_batch
            discount_batch = (1.0 - done_batch)  * self.gamma * gae_lambda
            gae = 0.0
            for i in reversed(range(reward_batch.size(0))):
                if done_batch[i]:
                    gae = reward_batch[i] - value_batch[i]
                elif truncated_batch[i]:
                    gae = delta_batch[i]
                else:
                    gae = delta_batch[i] + discount_batch[i] * gae
                advantage_batch[i] = gae
            return_batch = advantage_batch + value_batch

        else:
            raise NotImplementedError

        #normalize advantage
        if self.normalize_advantage:
            advantage_batch =  (advantage_batch - advantage_batch.mean()) / (advantage_batch.std() + 1e-8)

        return advantage_batch, return_batch
    
    def update(self, data_batch: dict):
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch, truncated_batch = \
            itemgetter("obs", "action", "reward", "next_obs", "done", "truncated")(data_batch)
        
        advantage_batch, return_batch =  self.estimate_advantage(obs_batch, action_batch, reward_batch, next_obs_batch, done_batch, truncated_batch)

        
        log_prob_batch = itemgetter("log_prob")(self.policy_network.evaluate_actions(obs_batch, action_batch))
        
        # Train policy with a single step of gradient descent
        loss_policy = -(log_prob_batch * advantage_batch).mean()
        self.policy_optimizer.zero_grad()
        loss_policy.backward()
        self.policy_optimizer.step()

        # Train value function
        for i in range(self.train_v_iters):
            estimation = self.v_network(obs_batch)
            loss_v = ((estimation - return_batch)**2).mean()
            self.v_optimizer.zero_grad()
            loss_v.backward()
            self.v_optimizer.step()

        return {
            "loss/policy": loss_policy,
            "loss/v": loss_v
        }

    @torch.no_grad()
    def select_action(self, obs, deterministic=False):
        if len(obs.shape) in [1, 3]:
            obs = [obs]
        if type(obs) != torch.tensor:
            obs = torch.FloatTensor(np.array(obs)).to(util.device)
        action, log_prob = itemgetter("action", "log_prob")(self.policy_network.sample(obs, deterministic=deterministic))
        if self.discrete_action_space:
            action = action[0]
        return {
            'action': action.detach().cpu().numpy(),
            'log_prob' : log_prob
            }