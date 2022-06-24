
from abc import abstractmethod
import numpy as np
import torch
from unstable_baselines.common import util
from unstable_baselines.common.data_structure import *
import gym
import random
from collections import namedtuple
import warnings
from gym.spaces import Box, Discrete

from unstable_baselines.common import util, functional

Transition = namedtuple('Transition', ['obs', 'action', 'next_obs', 'reward', 'done'])

class BaseBuffer(object):

    def __init__(self):
        pass

    @abstractmethod  
    def add_traj(self, obs_list, action_list, next_obs_list, reward_list, done_list):
        pass
    
    @abstractmethod  
    def add_transition(self):
        pass

    @abstractmethod
    def sample_batch(self):
        pass


class ReplayBuffer(object):
    def __init__(self, obs_space, action_space, max_buffer_size = 1000000,  **kwargs):
        self.max_buffer_size = max_buffer_size
        self.curr = 0
        self.obs_space =  obs_space
        self.action_space = action_space
        self.obs_dim = obs_space.shape[0]
        if type(action_space) == gym.spaces.discrete.Discrete:
            self.action_dim = 1
            #action_dim = action_space.n
            self.discrete_action = True
        elif type(action_space) == gym.spaces.box.Box:
            self.action_dim = action_space.shape[0]
            self.discrete_action = False
        else:
            assert 0, "unsupported action type"

        self.obs_buffer = np.zeros((max_buffer_size, self.obs_dim))
        if self.discrete_action:
            self.action_buffer = np.zeros((max_buffer_size, )).astype(np.long)
        else:
            self.action_buffer = np.zeros((max_buffer_size, self.action_dim))
        self.next_obs_buffer = np.zeros((max_buffer_size,self.obs_dim))
        self.reward_buffer = np.zeros((max_buffer_size,))
        self.done_buffer = np.zeros((max_buffer_size,))
        self.max_sample_size = 0

    def clear(self):
        self.max_sample_size = 0
        self.curr = 0

    def add_traj(self, obs_list, action_list, next_obs_list, reward_list, done_list):
        for obs, action, next_obs, reward, done in zip(obs_list, action_list, next_obs_list, reward_list, done_list):
            self.add_transition(obs, action, next_obs, reward, done)
    
    def add_transition(self, obs, action, next_obs, reward, done):
        self.obs_buffer[self.curr] = obs
        self.action_buffer[self.curr] = action
        self.next_obs_buffer[self.curr] = next_obs
        self.reward_buffer[self.curr] = reward
        self.done_buffer[self.curr] = done
        
        #increase pointer
        self.curr = (self.curr + 1) % self.max_buffer_size
        self.max_sample_size = min(self.max_sample_size+1, self.max_buffer_size)


    def sample(self, batch_size, to_tensor = True, sequential=False, allow_duplicate=False):
        if not allow_duplicate:
            if batch_size > self.max_sample_size:
                warnings.warn("Sampling larger than buffer size")
            batch_size = min(self.max_sample_size, batch_size)

        if sequential:
            start_index = random.choice(range(self.max_sample_size))
            indices = []
            for i in range(batch_size):
                indices.append( (start_index + i) % self.max_sample_size)
        elif allow_duplicate:
            indices = np.random.choice(range(self.max_sample_size), batch_size)
        else:
            indices = random.sample(range(self.max_sample_size), batch_size)
        
        obs_batch, action_batch, next_obs_batch, reward_batch, done_batch = self.obs_buffer[indices], \
            self.action_buffer[indices],\
            self.next_obs_buffer[indices],\
            self.reward_buffer[indices],\
            self.done_buffer[indices]
        if to_tensor:
            obs_batch = torch.FloatTensor(obs_batch).to(util.device)
            if self.discrete_action:
                action_batch = torch.LongTensor(action_batch).to(util.device)
            else:
                action_batch = torch.FloatTensor(action_batch).to(util.device)
            next_obs_batch = torch.FloatTensor(next_obs_batch).to(util.device)
            reward_batch = torch.FloatTensor(reward_batch).to(util.device).unsqueeze(1)
            done_batch = torch.FloatTensor(done_batch).to(util.device).unsqueeze(1)
        return dict( 
            obs=obs_batch, 
            action=action_batch, 
            next_obs=next_obs_batch, 
            reward=reward_batch, 
            done=done_batch
        )
        
    def get_batch(self, indices, to_tensor=True):
        obs_batch, action_batch, next_obs_batch, reward_batch, done_batch = self.obs_buffer[indices], \
            self.action_buffer[indices],\
            self.next_obs_buffer[indices],\
            self.reward_buffer[indices],\
            self.done_buffer[indices]
        if to_tensor:
            obs_batch = torch.FloatTensor(obs_batch).to(util.device)
            if self.discrete_action:
                action_batch = torch.LongTensor(action_batch).to(util.device)
            else:
                action_batch = torch.FloatTensor(action_batch).to(util.device)
            next_obs_batch = torch.FloatTensor(next_obs_batch).to(util.device)
            reward_batch = torch.FloatTensor(reward_batch).to(util.device).unsqueeze(1)
            done_batch = torch.FloatTensor(done_batch).to(util.device).unsqueeze(1)
        return dict( 
            obs=obs_batch, 
            action=action_batch, 
            next_obs=next_obs_batch, 
            reward=reward_batch, 
            done=done_batch
        )

    def resize(self, new_size):
        if new_size < self.max_buffer_size:
            self.obs_buffer = self.obs_buffer[:new_size]
            self.action_buffer = self.action_buffer[:new_size]
            self.next_obs_buffer =self.next_obs_buffer[:new_size]
            self.reward_buffer = self.reward_buffer[:new_size]
            self.done_buffer = self.done_buffer[:new_size]
            if self.curr >= new_size: #buffer has overflowed
                self.curr = 0
                self.max_sample_size = new_size
            self.max_buffer_size = new_size
        elif new_size == self.max_buffer_size:
            return
        elif new_size > self.max_buffer_size:
            addition_size = new_size - self.max_buffer_size
            
            #concatenate addition buffer to end
            new_obs_buffer = np.zeros((addition_size, self.obs_dim))
            if self.discrete_action:
                new_action_buffer = np.zeros((addition_size, )).astype(np.long)
            else:
                new_action_buffer = np.zeros((addition_size, self.action_dim))
                new_next_obs_buffer = np.zeros((addition_size,self.obs_dim))
                new_reward_buffer = np.zeros((addition_size,))
                new_done_buffer = np.zeros((addition_size,))
                self.obs_buffer = np.concatenate([self.obs_buffer, new_obs_buffer], axis=0)
                self.action_buffer = np.concatenate([self.action_buffer, new_action_buffer], axis=0)
                self.next_obs_buffer = np.concatenate([self.next_obs_buffer, new_next_obs_buffer], axis=0)
                self.reward_buffer = np.concatenate([self.reward_buffer, new_reward_buffer], axis=0)
                self.done_buffer = np.concatenate([self.done_buffer, new_done_buffer], axis=0)

                if self.curr < self.max_sample_size: #buffer has overflowed
                    self.curr = self.max_sample_size
                
            #update parameters:
            self.max_buffer_size = new_size


    def print_buffer_helper(self, nme, lst, summarize=False, print_curr_ptr = False):
        #for test purpose
        str_to_print = ""
        for i in range(self.max_sample_size):
            if print_curr_ptr:
                str_to_print += "^\t" if self.curr - 1 == i else "\t"  
            elif summarize:
                str_to_print += "{:.02f}\t".format(np.mean(lst[i]))
            else:
                str_to_print += "{:.02f}\t".format(lst[i])
        print("{}:\t{}" .format(nme, str_to_print))

    def print_buffer(self):
        #for test purpose
        self.print_buffer_helper("o",self.obs_buffer, summarize=True)
        #self.print_buffer_helper("a",self.action_buffer, summarize=True)
        self.print_buffer_helper("no",self.next_obs_buffer, summarize=True)
        self.print_buffer_helper("nxt_o",self.n_step_obs_buffer, summarize=True)
        self.print_buffer_helper("r",self.reward_buffer, summarize=True)
        self.print_buffer_helper("dis_r",self.discounted_reward_buffer, summarize=True)
        self.print_buffer_helper("done",self.done_buffer, summarize=True)
        self.print_buffer_helper("nxt_d",self.n_step_done_buffer, summarize=True)
        self.print_buffer_helper("n_count",self.n_count_buffer, print_curr_ptr=True)
        self.print_buffer_helper("indices", None, print_curr_ptr=True)
        print("\n")


class OnlineBuffer(object):
    """ Rollout Buffer for on-policy agent

    Base Args
    ----
    observation_space: gym.Space

    action_space: gym.Space

    kwargs Args
    -----------
    gamma: float
        Discount factor.

    size: int
        Buffer size

    max_trajectory_length: int
        Maximum length of a trajectory. It have to be consistent with the same value 
        `max_trajectory_length` in the Trainer.

    normalize_advantage: bool
        Whether use the advantage normalization trick.

    advantage_type: str, optional("gae")
        The advantage value type.

    gae_lambda: float
        Only work when advantage_type is "gae". The `\lambda` in GAE-Advantage.
    """
    def __init__(self,
            observation_space: gym.Space,
            action_space: gym.Space, 
            size: int,
            gamma: float,
            advantage_type: str,
            max_trajectory_length: int,
            normalize_advantage: bool,
            gae_lambda: float,
             **kwargs
             ):
        if isinstance(observation_space, Box):
            self.obs_dim = observation_space.shape[0]
        elif isinstance(observation_space, Discrete):
            self.obs_dim = observation_space.n
        else:
            raise NotImplementedError

        if isinstance(action_space, Box):
            self.action_dim = action_space.shape[0]
        elif isinstance(action_space, Discrete):
            self.action_dim = action_space.n
        else:
            raise NotImplementedError

        self.size = size
        self.gamma = gamma
        assert advantage_type in ["gae"], NotImplementedError
        self.advantage_type = advantage_type
        self.normalize_advantage = normalize_advantage
        self.gae_lambda = gae_lambda
        # self.td_n = kwargs['td_n']
        self.max_trajectory_length = max_trajectory_length

        # initialize buffer
        self.obs_buffer = np.zeros((self.size, self.obs_dim), dtype=np.float32)
        self.action_buffer = np.zeros((self.size, self.action_dim), dtype=np.float32)
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
        Call this function at the end of a trajectory, or when one gets cut off 
        by an epoch ending. This looks back in the buffer to where the trajectory
        started, and uses rewards and value estimates from the whole trajectory to
        compute advantage estimates with some methods, as well as compute the 
        rewards-to-go for each obs, to use as the targets fot the value function.

        The "last_value" argument should be 0 if the trajectory ended because the 
        agent reached a terminal obs (died), and otherwise should be V(s_T), the
        value function estimated for the last obs. This allows us to bootstrap the 
        reward-to-go calculation to account for timesteps beyond the arbitrary 
        episode horizon (or epoch cutoff).
        """
        path_slice = slice(self.path_start_idx, self.curr)
        rews = np.append(self.reward_buffer[path_slice], last_value)
        vals = np.append(self.value_buffer[path_slice], last_value)
        # GAE
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.advantage_buffer[path_slice] = functional.discount_cum_sum(deltas, self.gamma * self.gae_lambda)
        # compute the reward-to-go, to be targets for the value function
        self.return_buffer[path_slice] = functional.discount_cum_sum(rews, self.gamma)[:-1]

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
            obs=self.obs_buffer,
            action=self.action_buffer,
            ret=self.return_buffer[:, np.newaxis],
            advantage=self.advantage_buffer[:, np.newaxis],
            log_prob=self.log_prob_buffer[:, np.newaxis]            
        )
        if to_tensor:
            for key, value in data.items():
                data[key] = torch.FloatTensor(value).to(util.device)
        return data

    def add_transition(self, obs, action, reward, value, log_prob):
        """
        Store one timestep of agent-environment interaction to the buffer.
        """
        assert self.curr < self.max_size
        self.obs_buffer[self.curr] = obs
        self.action_buffer[self.curr] = action
        self.reward_buffer[self.curr] = reward
        self.value_buffer[self.curr] = value
        self.log_prob_buffer[self.curr] = log_prob
        self.curr += 1

    def add_trajectory(self, obs_list, action_list, reward_list, value_list, log_prob_list):
        """
        Store one trajectory to the buffer.
        """
        for obs, action, reward, value, log_prob in zip(
            obs_list,
            action_list,
            reward_list,
            value_list,
            log_prob_list
        ):
            self.store_transition(obs, action, reward, value, log_prob)


class TDReplayBuffer(ReplayBuffer):
    def __init__(self, obs_space, action_space, n, gamma, max_buffer_size = 1000000, **kwargs):
        self.n = n # parameter for td(n)
        self.max_buffer_size = max_buffer_size
        self.gamma = gamma
        obs_dim = obs_space.shape[0]
        if type(action_space) == gym.spaces.discrete.Discrete:
            action_dim = 1
            self.discrete_action = True
        elif type(action_space) == gym.spaces.box.Box:
            action_dim = action_space.shape[0]
            self.discrete_action = False
        else:
            assert 0, "unsupported action type"
        self.obs_buffer = np.zeros((max_buffer_size, obs_dim))
        self.action_buffer = np.zeros((max_buffer_size, action_dim))
        self.next_obs_buffer = np.zeros((max_buffer_size,obs_dim))
        self.reward_buffer = np.zeros((max_buffer_size,))
        self.done_buffer = np.ones((max_buffer_size,))
        self.n_step_obs_buffer = np.zeros((max_buffer_size,obs_dim))
        self.discounted_reward_buffer = np.zeros((max_buffer_size,))
        self.n_step_done_buffer = np.zeros(max_buffer_size,)
        self.n_count_buffer = np.ones((max_buffer_size,)).astype(np.int) * self.n
        #insert a random state at initialization to avoid bugs when inserting the first state
        self.max_sample_size = 1
        self.curr = 1
    
    def add_transition(self, obs, action, next_obs, reward, done):
        # store to instant memories
        self.obs_buffer[self.curr] = obs
        self.action_buffer[self.curr] = action
        self.next_obs_buffer[self.curr] = next_obs
        self.reward_buffer[self.curr] = reward
        self.done_buffer[self.curr] = done
        #store precalculated tn(n) info
        self.n_step_obs_buffer[self.curr] = next_obs
        self.discounted_reward_buffer[self.curr] = reward
        self.n_step_done_buffer[self.curr] = 0.
        breaked = False # record if hit the previous trajectory
        for i in range(self.n - 1):
            idx = (self.curr - i - 1) % self.max_sample_size # use max sample size cuz the buffer might not have been full
            if self.done_buffer[idx]: # hit the previous trajecory, break
                breaked = True
                break
            self.discounted_reward_buffer[idx] += (self.gamma ** (i + 1))  * reward
        if not breaked  and not self.done_buffer[(self.curr - self.n) % self.max_sample_size]:# not hit last trajctory, set the n-step-next state for the last state
            self.n_step_obs_buffer[(self.curr - self.n) % self.max_sample_size] = obs
        if done:#set the n-step-next-obs of the previous n states to the current state
            self.n_step_done_buffer[self.curr] = 1.0 
            self.n_count_buffer[self.curr] = 1
            for i in range(self.n - 1):
                idx = (self.curr - i -1) % self.max_sample_size
                if self.done_buffer[idx]:# hit the last trajectory
                    break
                self.n_step_obs_buffer[idx] = next_obs
                self.n_step_done_buffer[idx] = 1.0
                self.n_count_buffer[idx] = i + 2
        else:
            prev_idx = (self.curr - 1) % self.max_sample_size
            if not self.done_buffer[prev_idx]:
                self.n_step_done_buffer[prev_idx] = 0.
            for i in range(self.n - 1):
                idx = (self.curr - i - 1) % self.max_sample_size
                if self.done_buffer[idx]:# hit the last trajectory
                    break
                self.n_step_obs_buffer[idx] = next_obs
                self.n_step_done_buffer[idx] = 0.0

        # another special case is that n > max_sample_size, that might casuse a cyclic visiting of a buffer that has no done states
        # this has been avoided by setting initializing all done states to true
        self.curr = (self.curr+1) % self.max_buffer_size
        self.max_sample_size = min(self.max_sample_size + 1, self.max_buffer_size)

    def sample_batch(self, batch_size, to_tensor=True, n=None):
        if n is not None and n != self.n:
            self.update_td(n)
        # compute valid indices to sample (in case the sampled td tuples are incorrect due to the done states)
        if self.done_buffer[self.curr - 1]:
            # whole trajectories have been recorded
            valid_indices =range(self.max_sample_size)
        elif self.curr >= self.n:
            #ignore the previous n tuples
            valid_indices = list(range(self.curr - self.n)) + list(range(self.curr + 1, self.max_sample_size))
        else:
            valid_indices = range(self.curr + 1, self.max_sample_size - (self.n - self.curr))
        batch_size = min(len(valid_indices), batch_size)
        indices = random.sample(valid_indices, batch_size)
        obs_batch, action_batch, n_step_obs_batch, discounted_reward_batch, n_step_done_batch =\
            self.obs_buffer[indices], \
            self.action_buffer[indices],\
            self.n_step_obs_buffer[indices],\
            self.discounted_reward_buffer[indices],\
            self.n_step_done_buffer[indices]
        if to_tensor:
            obs_batch = torch.FloatTensor(obs_batch).to(util.device)
            if self.discrete_action:
                action_batch = torch.LongTensor(action_batch).to(util.device)
            else:
                action_batch = torch.FloatTensor(action_batch).to(util.device)
            n_step_obs_batch = torch.FloatTensor(n_step_obs_batch).to(util.device)
            discounted_reward_batch = torch.FloatTensor(discounted_reward_batch).to(util.device).unsqueeze(1)
            n_step_done_batch = torch.FloatTensor(n_step_done_batch).to(util.device).unsqueeze(1)
            
        return obs_batch, action_batch, n_step_obs_batch, discounted_reward_batch, n_step_done_batch

    def update_td(self, n):
        # function: update the current buffer to the new td(n) mode
        print("Updating the current buffer from td \033[32m{} to {}\033[0m".format(self.n, n))
        # reset all discounted_reward_batch, n_step_done_batch and n_step_obs_batch
        self.n_step_obs_buffer = np.zeros_like(self.n_step_obs_buffer)
        self.discounted_reward_buffer = np.zeros_like(self.discounted_reward_buffer)
        self.n_step_done_buffer = np.zeros_like(self.n_step_done_buffer)
        self.mask_buffer = np.zeros_like(self.n_step_done_buffer)
        curr = (self.curr - 1) % self.max_sample_size # self.curr points to the index to be overwrite, so decrease by 1
        curr_traj_end_idx = curr # mark the end of the current trajectory
        num_trajs = int(np.sum(self.done_buffer))
        if not self.done_buffer[curr]:
            num_trajs += 1
        while num_trajs > 0:
            self.n_step_done_buffer[curr_traj_end_idx] = self.done_buffer[curr_traj_end_idx] # set done of the last state
            self.n_step_obs_buffer[curr_traj_end_idx] = self.next_obs_buffer[curr_traj_end_idx] # set the next obs of the last state
            #calculate the length of the current trajectory
            curr_traj_len = 1
            idx = (curr_traj_end_idx - 1) % self.max_sample_size
            while(not self.done_buffer[idx] and idx != curr):
                idx = (idx - 1) % self.max_sample_size
                curr_traj_len += 1
            #backward through the last n states and set the n step done/obs buffer
            for i in range( min (n - 1, curr_traj_len)):
                idx = (curr_traj_end_idx - i - 1) % self.max_sample_size
                if self.done_buffer[idx]:
                    break
                self.n_step_obs_buffer[idx] = self.next_obs_buffer[curr_traj_end_idx]
                self.n_step_done_buffer[idx] = self.done_buffer[curr_traj_end_idx]
            #accumulate the discounted return
            for i in range(curr_traj_len):
                curr_return = self.reward_buffer[ (curr_traj_end_idx - i) % self.max_sample_size]
                for j in range( min(n, curr_traj_len - i)):
                    target_idx = curr_traj_end_idx - i - j 
                    self.discounted_reward_buffer[target_idx] += (curr_return * (self.gamma ** j))
            # set the n_step_done/obs buffer
            if  curr_traj_len >= n:
                for i in range(curr_traj_len - n): # 3
                    curr_idx = (curr_traj_end_idx - n - i) % self.max_sample_size
                    if self.done_buffer[curr_idx]:
                        break
                    next_obs_idx = (curr_idx + n) % self.max_sample_size
                    self.n_step_obs_buffer[curr_idx] = self.obs_buffer[next_obs_idx]
            curr_traj_end_idx = (curr_traj_end_idx - curr_traj_len) % self.max_sample_size
            num_trajs -= 1
            
        self.n = n

class PrioritizedReplayBuffer(BaseBuffer):
    def __init__(self, obs_space, action_space, max_buffer_size=1000000, metric='propotional', **kargs):
        self.max_buffer_size = max_buffer_size
        self.args = kargs
        self.curr = 0
        obs_dim = obs_space.shape[0]
        if type(action_space) == gym.spaces.discrete.Discrete:
            action_dim = 1
            self.discrete_action = True
        elif type(action_space) == gym.spaces.box.Box:
            action_dim = action_space.shape[0]
            self.discrete_action = False
        else:
            raise TypeError("Type of action must be either Discrete or Box!")

        if type(metric) == str:
            if metric == 'propotional':
                self.buffer = SumTree(self.max_buffer_size)
                self.metric_fn = self._propotional
                self.alpha = self.args["alpha"]
                self.beta = self.args["init_beta"]
                self.final_beta = self.args["final_beta"]
                self.beta_decay = self.args["beta_decay"]
                self.epsilon = self.args["epsilon"] 
            elif metric == 'rank':
                raise NotImplementedError
            else:
                raise NotImplementedError("Built-in metrics for PER are propotional and rank")
        elif type(metric) == function:
            self.metric_fn = metric
        else:
            raise TypeError("Metric should be either str (use built-in metric) or function (use custom function)")

        self.max_sample_size = 0

    def add_traj(self, obs_list, action_list, next_obs_list, reward_list, done_list, metric_list):
        for obs, action, next_obs, reward, done, metric in zip(obs_list, action_list, next_obs_list, reward_list, done_list, metric_list):
            self.add_transition(obs, action, next_obs, reward, done, metric)

    def add_transition(self, obs, action, next_obs, reward, done, metric):
        metric = self.metric_fn(metric)
        t = Transition(obs, action, next_obs, reward, done)
        self.buffer.add(metric, t)
        
        self.curr = (self.curr+1)%self.max_buffer_size
        self.max_sample_size = min(self.max_sample_size+1, self.max_buffer_size)

    def sample_batch(self, batch_size, to_tensor=True):
        batch_size = min(batch_size, self.max_sample_size)
        trans_batch = []
        IS_batch = []
        info_batch = []
        total_p = self.buffer.value[0]
        segment = 1/batch_size
        for i_seg in range(batch_size):
            target = (random.random()+i_seg) * segment
            idx, p, transition = self.buffer.find(target)
            trans_batch.append(transition)
            IS_batch.append(np.power((self.buffer.size*p/total_p), -self.beta))
            info_batch.append(idx)
        obs_batch, action_batch, next_obs_batch, reward_batch, done_batch = \
            [i.obs for i in trans_batch], \
            [i.action for i in trans_batch], \
            [i.next_obs for i in trans_batch], \
            [i.reward for i in trans_batch], \
            [i.done for i in trans_batch]
        
        if to_tensor:
            obs_batch = torch.FloatTensor(obs_batch).to(util.device)
            if self.discrete_action:
                action_batch = torch.LongTensor(action_batch).to(util.device)
            else:
                action_batch = torch.FloatTensor(action_batch).to(util.device)
            next_obs_batch = torch.FloatTensor(next_obs_batch).to(util.device)
            reward_batch = torch.FloatTensor(reward_batch).to(util.device).unsqueeze(1)
            done_batch = torch.FloatTensor(done_batch).to(util.device).unsqueeze(1)
            IS_batch = torch.FloatTensor(IS_batch)
            IS_batch = (IS_batch/IS_batch.max()).to(util.device).unsqueeze(1)
            info_batch = torch.FloatTensor(info_batch).to(util.device)

        # decay beta
        self.beta = min(self.beta + self.beta*self.beta_decay, self.final_beta)

        return obs_batch, action_batch, next_obs_batch, reward_batch, done_batch, IS_batch, info_batch
    
    def batch_update(self, idxs, errors):
        for idx, error in zip(idxs, errors):
            self.buffer.update(idx, self.metric_fn(error))

    @property
    def max(self):
        return self.buffer.max

    def __str__ (self):
        return self.buffer.__str__()

    def _propotional(self, metric):
        return (np.abs(metric) + self.epsilon) ** self.alpha
        
    
if __name__ == "__main__":
    from tqdm import tqdm
    
    
    #code for testing discrete action environments
    # env = gym.make("CartPole-v0")
    # obs_space = env.observation_space
    # action_space = env.action_space
    # buffer = ReplayBuffer(obs_space, action_space, max_buffer_size=10)
    # for traj  in tqdm(range(50)):
    #     done = False
    #     obs = env.reset()
    #     while not done:
    #         action = action_space.sample()
    #         next_obs,  reward, done, _ = env.step(action)
    #         buffer.add_transition(obs, action, next_obs, reward, done)
    #         obs = next_obs

    # code for testing normal buffer
    # env = gym.make("HalfCheetah-v2")
    # obs_space = env.observation_space
    # action_space = env.action_space
    # buffer = ReplayBuffer(obs_space, action_space, max_buffer_size=10)
    # 
    # for traj  in tqdm(range(50)):
    #     done = False
    #     obs = env.reset()
    #     while not done:
    #         action = action_space.sample()
    #         next_obs,  reward, done, _ = env.step(action)
    #         buffer.add_transition(obs, action, next_obs, reward, done)
    #         obs = next_obs
    # obs_batch, action_batch, next_obs_batch, reward_batch, done_batch = buffer.sample(32,)
    # print(obs_batch[0].shape, action_batch[0].shape, next_obs_batch[0].shape, reward_batch[0].shape, done_batch[0].shape)
    # obs_batch, action_batch, next_obs_batch, reward_batch, done_batch = buffer.sample(32, step_size = 2)
    # print(obs_batch[0].shape, action_batch[0].shape, next_obs_batch[0].shape, reward_batch[0].shape, done_batch[0].shape)
    # obs_batch, action_batch, next_obs_batch, reward_batch, done_batch = buffer.sample(32, step_size = 5)
    # print(obs_batch[0].shape, action_batch[0].shape, next_obs_batch[0].shape, reward_batch[0].shape, done_batch[0].shape)
    # obs_batch, action_batch, next_obs_batch, reward_batch, done_batch = buffer.sample(32, step_size = -1)
    # print(obs_batch[0].shape, action_batch[0].shape, next_obs_batch[0].shape, reward_batch[0].shape, done_batch[0].shape)


    #code for testing td buffer
    env = gym.make("HalfCheetah-v2")
    #env = gym.make("CartPole-v1")
    obs_space = env.observation_space
    action_space = env.action_space
    alpha = 0.8
    max_buffer_size = 16
    max_traj_length = 5
    num_trajs = 4
    ER = PrioritizedReplayBuffer(obs_space, action_space, max_buffer_size, alpha=alpha, epsilon=0.01)

    for traj in tqdm(range(num_trajs)):
        done = False
        obs = env.reset()
        num_steps = 0
        while not done:
            action = action_space.sample()
            next_obs, reward, done, _ = env.step(action)
            num_steps += 1
            if num_steps > max_traj_length:
                done = True
            ER.add_transition(obs, action, next_obs, reward, done, np.random.random()*10)
            obs = next_obs
            print("step! ----------------------------------------")
            print(ER)

    print("==============================================")
    for i in tqdm(range(4)):
        _, _, _, _, _, p_batch = ER.sample_batch(4, False)
        print(p_batch)

            