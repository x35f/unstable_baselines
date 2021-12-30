import gym
from gym.spaces import Box, Discrete
import torch
import random
import numpy as np

from unstable_baselines.common import util
from unstable_baselines.common import util
from unstable_baselines.common.buffer import ReplayBuffer, TDReplayBuffer


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
    def __init__(self,
                 state_space: gym.Space,
                 action_space: gym.Space, 
                 **kwargs):
        """
        Args
        ----
        state_space: gym.Space

        action_space: gym.Space

        size: int = 1000
            Buffer size.

        gamma: float = 0.99
            Discount factor.
    
        advantage_type: str ["gae"]
            Determine the advantage function computation model.
            "gae": Generalized Advantage estimation
        
        normalize_advantage: bool = True

        gae_lambda: float = 0.95
            Lambda for GAE-Lambda.

        max_ep_length: int = 1000
            Maximum length of trajectory / episode / rollout.

        td_n: int = 1
            N for TD(n).

        """
        
        """
        
        在spinning up中的rollout buffer的函数

        def store()
            保存一步transition到buffer当中

        def finish_path(self, )
            在轨迹结束后调用
            last_val argument 在轨迹结束的时候应该是0，因为agent到达了一个终止状态，
            计算 adv_buf, ret_buf

        def get(self)
            在epoch结束的时候调用，
        
        
        """

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
        # TODO(mimeku): 看下具体的normalization trick到底是什么样的
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



    # def finish_traj(self, value_network):
    #     path_slice = slice(self.path_start_idx, self.curr)
    #     obs = torch.FloatTensor(self.obs_buffer[path_slice]).to(util.device)
    #     values = value_network(obs).detach().cpu().numpy().flatten()
    #     if self.done_buffer[self.curr - 1]:
    #         last_val = 0.
    #     else:#timeout case
    #         last_obs = torch.Tensor([self.next_obs_buffer[self.curr - 1]])
    #         last_val = value_network(last_obs).detach().cpu().numpy()[0]
    #     rews = np.append(self.reward_buffer[path_slice], last_val)
    #     values = np.append(values, last_val)
    #     # the next two lines implement GAE-Lambda advantage calculation
    #     deltas = rews[:-1] + self.gamma * values[1:] - values[:-1]
    #     self.advantage_buffer[path_slice] = util.discount_cum_sum(deltas, self.gamma * self.gae_lambda)
        
    #     # the next line computes rewards-to-go, to be targets for the value function
    #     self.return_buffer[path_slice] = util.discount_cum_sum(rews, self.gamma)[:-1]
    #     self.path_start_idx = self.curr

    # def reset(self):
    #     #delete buffers
    #     del self.obs_buffer, self.action_buffer, self.log_pi_buffer, self.reward_buffer, self.done_buffer,\
    #          self.next_obs_buffer, self.return_buffer, self.advantage_buffer#, self.value_buffer

    #     self.obs_buffer = np.zeros((self.max_buffer_size, self.obs_dim)).astype(float)
    #     self.next_obs_buffer = np.zeros((self.max_buffer_size, self.obs_dim)).astype(float)
    #     self.action_buffer = np.zeros((self.max_buffer_size, self.action_dim))
    #     self.log_pi_buffer = np.zeros((self.max_buffer_size,)).astype(float)
    #     self.reward_buffer = np.zeros((self.max_buffer_size,)).astype(float)
    #     self.done_buffer = np.zeros((self.max_buffer_size,)).astype(float)
    #     self.return_buffer = np.zeros((self.max_buffer_size,)).astype(float)# calculate return for the whole trajectory
    #     self.advantage_buffer = np.zeros((self.max_buffer_size,)).astype(float)# calculate return for the whole trajectory
    #     self.max_sample_size = 0
    #     self.curr = 0
    #     self.path_start_idx = 0
    #     self.finalized = False

    # def add_traj(self, obs_list, action_list, log_pi_list, next_obs_list, reward_list, done_list, ):
    #     for obs, action, log_pi, next_obs, reward, done  in zip(obs_list, action_list, log_pi_list, next_obs_list, reward_list, done_list):
    #         self.add_tuple(obs, action, log_pi, next_obs, reward, done)
    
    # def add_tuple(self, obs, action, log_pi, next_obs, reward, done):
    #     # store to instant memories
    #     self.obs_buffer[self.curr] = np.array(obs).copy()
    #     self.action_buffer[self.curr] = np.array(action)
    #     self.log_pi_buffer[self.curr] = np.array(log_pi)
    #     self.next_obs_buffer[self.curr] = np.array(next_obs)
    #     self.reward_buffer[self.curr] = np.array(reward)
    #     self.done_buffer[self.curr] = np.array(done)
    #     self.curr = (self.curr+1) % self.max_buffer_size
    #     self.max_sample_size = min(self.max_sample_size + 1, self.max_buffer_size)

    
    # def finish_path(self):
    #     """
    #     """
    #     pass
    
    # def finalize(self, value_network):
    #     self.obs_buffer = torch.FloatTensor(self.obs_buffer[:self.max_sample_size]).to(util.device)
    #     self.next_obs_buffer = torch.FloatTensor(self.next_obs_buffer[:self.max_sample_size]).to(util.device)
    #     #convert the remaining buffers to tensor and pass data to device
    #     self.reward_buffer = torch.FloatTensor(self.reward_buffer[:self.max_sample_size]).to(util.device)
    #     #self.value_buffer = torch.FloatTensor(self.value_buffer[:self.max_sample_size]).to(util.device).unsqueeze(1)
    #     self.action_buffer = torch.FloatTensor(self.action_buffer[:self.max_sample_size]).to(util.device)
    #     self.log_pi_buffer = torch.FloatTensor(self.log_pi_buffer[:self.max_sample_size]).to(util.device).unsqueeze(1)
    #     self.advantage_buffer = torch.FloatTensor(self.advantage_buffer[:self.max_sample_size]).to(util.device).unsqueeze(1)
    #     self.return_buffer = torch.FloatTensor(self.return_buffer[:self.max_sample_size]).to(util.device).unsqueeze(1)
    #     #self.done_buffer = torch.FloatTensor(self.done_buffer[:self.max_sample_size]).to(util.device).unsqueeze(1)
    #     self.finalized = True

    # def sample_batch(self, batch_size, to_tensor = True, step_size: int = 1):
    #     # batch_size: 
    #     # to_tensor: if convert to torch.tensor type as pass to util.device
    #     # step_size: return a list of next states, returns and dones with size n
    #     if not self.finalized:
    #         print("sampling before finalizing the buffer")
    #         assert 0
        
    #     batch_size = min(self.max_sample_size, batch_size)
    #     index = random.sample(range(self.max_sample_size), batch_size)
    #     obs_batch, action_batch, log_pi_batch, next_obs_batch, reward_batch, advantage_batch, return_batch, done_batch = \
    #         self.obs_buffer[index], \
    #         self.action_buffer[index],\
    #         self.log_pi_buffer[index],\
    #         self.next_obs_buffer[index],\
    #         self.reward_buffer[index],\
    #         self.advantage_buffer[index],\
    #         self.return_buffer[index],\
    #         self.done_buffer[index]
    #     return dict(
    #         obs=obs_batch, 
    #         action=action_batch, 
    #         log_pi=log_pi_batch,
    #         next_obs=next_obs_batch, 
    #         reward=reward_batch, 
    #         advantage=advantage_batch, 
    #         ret=return_batch, 
    #         done=done_batch
    #     )


    # def print_buffer_helper(self, nme, lst, summarize=False, print_curr_ptr = False):
    #     if type(lst) == torch.Tensor:
    #         lst = lst.detach().cpu().numpy()
    #     #for test purpose
    #     #print(type(lst), self.max_sample_size)
    #     str_to_print = ""
    #     for i in range(self.max_sample_size):
    #         if print_curr_ptr:
    #             str_to_print += "^\t" if self.curr - 1 == i else "\t"  
    #         elif summarize:
    #             str_to_print += "{:.02f}\t".format(np.mean(lst[i]))
    #         else:
    #             str_to_print += "{:.02f}\t".format(lst[i])
    #     print("{}:\t{}" .format(nme, str_to_print))

    # def print_buffer(self):
    #     #for test purpose
    #     self.print_buffer_helper("o",self.obs_buffer, summarize=True)
    #     #self.print_buffer_helper("a",self.action_buffer, summarize=True)
    #     self.print_buffer_helper("nxt_o",self.next_obs_buffer, summarize=True)
    #     self.print_buffer_helper("r",self.reward_buffer, summarize=True)
    #     self.print_buffer_helper("ret",self.return_buffer, summarize=True)
    #     self.print_buffer_helper("value",self.value_buffer, summarize=True)
    #     self.print_buffer_helper("adv",self.advantage_buffer, summarize=True)
    #     self.print_buffer_helper("done",self.done_buffer, summarize=True)
    #     self.print_buffer_helper("index", None, print_curr_ptr=True)
    #     print("\n")



    
    

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