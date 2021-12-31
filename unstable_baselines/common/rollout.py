import gym
from gym.spaces import Box, Discrete
import torch
import random
import numpy as np

from unstable_baselines.common import util


def rollout_trajectory(env, agent, max_traj_length):
    states, actions, log_pis, next_states, rewards, dones = [], [], [], [], [], []
    state = env.reset()
    done = False
    traj_length = 0
    while not done:
        action, log_pi = agent.select_action(state)
        #clipped_action = np.clip(action, env.action_space.low, env.action_space.high)
        #next_state, reward, done, info = env.step(clipped_action)
        next_state, reward, done, info = env.step(action)
        traj_length += 1
        timed_out = traj_length >= max_traj_length
        states.append(state)
        actions.append(action)
        log_pis.append(log_pi.item())
        next_states.append(next_state)
        rewards.append(reward)
        dones.append(done)
        state = next_state
        if done or traj_length >= max_traj_length:
            break
    return states, actions, log_pis, next_states, rewards, dones


class RolloutBuffer(object):
    """ Rollout Buffer for on-policy agent

    Base Args
    ----
    state_space: gym.Space

    action_space: gym.Space

    kwargs Args
    -----------
    gamma: float
        Discount factor.

    size: int
        Buffer size

    max_ep_length: int
        Maximum length of a trajectory. It have to be consistent with the same value 
        `max_ep_length` in the Trainer.

    normalize_advantage: bool
        Whether use the advantage normalization trick.

    advantage_type: str, optional("gae")
        The advantage value type.

    gae_lambda: float
        Only work when advantage_type is "gae". The `\lambda` in GAE-Advantage.
    """
    def __init__(self,
                 state_space: gym.Space,
                 action_space: gym.Space, 
                 **kwargs):
        if isinstance(state_space, Box):
            self.state_dim = state_space.shape[0]
        elif isinstance(state_space, Discrete):
            self.state_dim = state_space.n
        else:
            raise NotImplementedError

        if isinstance(action_space, Box):
            self.action_dim = action_space.shape[0]
        elif isinstance(action_space, Discrete):
            self.action_dim = action_space.n
        else:
            raise NotImplementedError

        self.size = kwargs['size']
        self.gamma = kwargs['gamma']
        assert kwargs['advantage_type'] in ["gae"], NotImplementedError
        self.advantage_type = kwargs['advantage_type']
        self.normalize_advantage = kwargs['normalize_advantage']
        self.gae_lambda = kwargs['gae_lambda']
        # self.td_n = kwargs['td_n']
        self.max_ep_length = kwargs['max_ep_length']

        # initialize buffer
        self.state_buffer = np.zeros(util.combine_shape(self.size, self.state_dim), dtype=np.float32)
        self.action_buffer = np.zeros(util.combine_shape(self.size, self.action_dim), dtype=np.float32)
        self.advantage_buffer = np.zeros(self.size, dtype=np.float32)
        self.reward_buffer = np.zeros(self.size, dtype=np.float32)
        self.return_buffer = np.zeros(self.size, dtype=np.float32)
        self.value_buffer = np.zeros(self.size, dtype=np.float32)
        self.log_prob_buffer = np.zeros(self.size, dtype=np.float32)
        
        # initialize pointer
        self.curr = 0
        self.path_start_idx = 0
        self.max_size = self.size

    def finish_path(self, last_value=0):
        """
        Call this funciton at the end of a trajectory, or when one gets cut off 
        by an epoch ending. This looks back in the buffer to where the trajectory
        started, and uses rewards and value estimates from the whole trajectory to
        compute advantage estimates with some methods, as well as compute the 
        rewards-to-go for each state, to use as the targets fot the value function.

        The "last_value" argument should be 0 if the trajectory ended because the 
        agent reached a terminal state (died), and otherwise should be V(s_T), the
        value function estimated for the last state. This allows us to bootstrap the 
        reward-to-go calculation to account for timesteps beyond the arbitrary 
        episode horizon (or epoch cutoff).
        """
        path_slice = slice(self.path_start_idx, self.curr)
        rews = np.append(self.reward_buffer[path_slice], last_value)
        vals = np.append(self.value_buffer[path_slice], last_value)
        # GAE
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.advantage_buffer[path_slice] = util.discount_cum_sum(deltas, self.gamma * self.gae_lambda)
        # compute the reward-to-go, to be targets for the value function
        self.return_buffer[path_slice] = util.discount_cum_sum(rews, self.gamma)[:-1]

        self.path_start_idx = self.curr

    def get(self, to_tensor=True):
        """
        Call this at the end of an epoch to get all of the data from the buffer,
        with advantages appropriately normalized (shifted to have mean zero and std one).
        Also, resets some pointers in the buffer.

        Returns
        -------
        data: dict
            obs, act, ret, adv, logp
        """
        assert self.curr == self.max_size
        self.curr, self.path_start_idx = 0, 0
        # normalize the advantage function
        if self.normalize_advantage:
            adv_mean = np.mean(self.advantage_buffer)
            adv_std = np.std(self.advantage_buffer)
            self.advantage_buffer = (self.advantage_buffer - adv_mean) / adv_std
        data = dict(
            obs=self.state_buffer,
            act=self.action_buffer,
            ret=self.return_buffer[:, np.newaxis],
            adv=self.advantage_buffer[:, np.newaxis],
            logp=self.log_prob_buffer[:, np.newaxis]            
        )
        if to_tensor:
            for key, value in data.items():
                data[key] = torch.FloatTensor(value).to(util.device)
        return data

    def store_transition(self, state, action, reward, value, log_prob):
        """
        Store one timestep of agent-environment interaction to the buffer.
        """
        assert self.curr < self.max_size
        self.state_buffer[self.curr] = state
        self.action_buffer[self.curr] = action
        self.reward_buffer[self.curr] = reward
        self.value_buffer[self.curr] = value
        self.log_prob_buffer[self.curr] = log_prob
        self.curr += 1

    def store_trajectory(self, state_list, action_list, reward_list, value_list, log_prob_list):
        """
        Store one trajectory to the buffer.
        """
        for state, action, reward, value, log_prob in zip(
            state_list,
            action_list,
            reward_list,
            value_list,
            log_prob_list
        ):
            self.store_transition(state, action, reward, value, log_prob)

    def store(self, state, action, reward, value, log_prob):
        """
        Simplify self.store_transition.
        """
        self.store_transition(state, action, reward, value, log_prob)


if __name__ == "__main__":
    from tqdm import tqdm
    import gym
    from unstable_baselines.common.agents import RandomAgent


    #code for testing td buffer
    env = gym.make("HalfCheetah-v2")
    #env = gym.make("CartPole-v1")
    obs_space = env.observation_space
    action_space = env.action_space
    n = 1
    gamma = 0.5
    max_buffer_size = 12
    max_traj_length = 5
    num_trajs = 4
    agent = RandomAgent(obs_space, action_space)
    rollout_buffer = RolloutBuffer(obs_space=obs_space,
                        action_space=action_space,
                        max_trajectory_length=max_traj_length,
                        gamma=gamma,
                        n=n,
                        max_env_steps=max_buffer_size,
                        advantage_type="gae")
    
    l, r = rollout_buffer.collect_trajectories(env, agent,n=n)
    rollout_buffer.print_buffer()
