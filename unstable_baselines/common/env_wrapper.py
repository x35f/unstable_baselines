import gym
import numpy as np
from copy import deepcopy
from collections import deque
# from mujoco_py import GlfwContext
# GlfwContext(offscreen=True) 
from gym.envs.registration import register
from gym.spaces.box import Box
from PIL import Image
import cv2
import random
from gym import spaces
from gym import core, spaces

from dm_env import specs

MUJOCO_SINGLE_ENVS = [
    'Ant-v2', 'Ant-v3', 'Ant-v4',
    'HalfCheetah-v2', 'HalfCheetah-v3','HalfCheetah-v4',
    'Hopper-v2', 'Hopper-v3',  'Hopper-v4',
    'Humanoid-v2', 'Humanoid-v3', 'Humanoid-v4',
    'InvertedDoublePendulum-v2', 'InvertedDoublePendulum-v4',
    'InvertedPendulum-v2', 'InvertedPendulum-v3',  'InvertedPendulum-v4',
    'Swimmer-v2', 'Swimmer-v3', 'Swimmer-v4',
    'Walker2d-v2', 'Walker2d-v3', 'Walker2d-v4',
    'Pusher-v2',
    'Reacher-v2',
    'Striker-v2',
    'Thrower-v2',
    'CartPole-v1',
    'MountainCar-v0'
    ]
METAWORLD_META_ENVS = [
    'ML1', 'ML10', 'ML45'
]
METAWORLD_SINGLE_ENVS = [   
    'reach-v2', 'bin-picking-v2', 'handle-press-v2', 'handle-pull-side-v2', 'push-back-v2', 
    'stick-pull-v2', 'sweep-into-v2', 'button-press-v2', 'reach-wall-v2', 'window-close-v2', 
    'hammer-v2', 'button-press-topdown-v2', 'handle-pull-v2', 'pick-out-of-hole-v2', 
    'coffee-pull-v2', 'door-close-v2', 'pick-place-wall-v2', 'coffee-push-v2', 'drawer-close-v2', 'dial-turn-v2', 'peg-unplug-side-v2', 'assembly-v2', 'button-press-wall-v2', 'peg-insert-side-v2', 
    'plate-slide-back-side-v2', 'drawer-open-v2', 'disassemble-v2', 'plate-slide-side-v2', 
    'door-lock-v2', 'push-v2', 'window-open-v2', 'hand-insert-v2', 'faucet-open-v2', 
    'plate-slide-v2', 'box-close-v2', 'basketball-v2', 'coffee-button-v2', 'door-open-v2', 
    'button-press-topdown-wall-v2', 'lever-pull-v2', 'shelf-place-v2', 'handle-press-side-v2', 
    'pick-place-v2', 'soccer-v2', 'push-wall-v2', 'stick-push-v2', 'door-unlock-v2', 'sweep-v2', 
    'plate-slide-back-v2', 'faucet-close-v2'
]
MUJOCO_META_ENVS = [
    'walker-rand-params', 
    'humanoid-dir', 'hopper-rand-params', 'ant-dir', 
    'cheetah-vel', 'cheetah-dir', 'ant-goal']

METAWORLD_ENVS = ['MetaWorld']

MBPO_ENVS = [
    'AntTruncatedObs-v2',
    'HumanoidTruncatedObs-v2',
    ]
ATARI_ENVS = [
    'Pong-v4', 'PongNoFrameskip-v4', 'QbertNoFrameskip-v4', 
    'BreakoutNoFrameskip-v4',"MsPacmanNoFrameskip-v4",
    "SeaquestNoFrameskip-v4", "BreakoutNoFrameskip-v4", 'Qbert-v4', 
    'Breakout-v4',"MsPacman-v4",
    "Seaquest-v4", "BeamRider-v4", "BeamRiderNoFrameskip-v4",
    "DemonAttack-v4", "SpaceInvaders-v4","SpaceInvadersNoFrameskip-v4",
    "TimePilot-v4",
]

DMC_ENVS = [
    "acrobot-swingup", "acrobot-swingup_sparse", "ball_in_cup-catch", "cartpole-balance", "cartpole-balance_sparse", "cartpole-swingup", "cartpole-swingup_sparse", "cheetah-run", "finger-spin", "finger-turn_easy", "finger-turn_hard", "fish-upright", "fish-swim", "hopper-stand", "hopper-hop", "humanoid-stand", "humanoid-walk", "humanoid-run", "manipulator-bring_ball", "pendulum-swingup", "point_mass-easy", "reacher-easy", "reacher-hard", "swimmer-swimmer6", "swimmer-swimmer15", "walker-stand", "walker-walk", "walker-run"
]

def get_env(env_name, seed=None, **kwargs):
    if seed is None:
        seed = np.random.randint(10000)
    if env_name in MUJOCO_SINGLE_ENVS:
        env = gym.make(env_name, **kwargs)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        return env
    elif env_name in MUJOCO_META_ENVS:
        if env_name in [ 'walker-rand-params', 'hopper-rand-params']:
            print("\033[31mwalker-rand-params and hopper-rand-params are currently not supported by the new version of mujoco and gym\033[0m")
            assert(0)
        from unstable_baselines.envs.mujoco_meta.rlkit_envs import ENVS as MUJOCO_META_ENV_LIB
        num_train_tasks = kwargs['num_train_tasks']
        num_eval_tasks = kwargs['num_eval_tasks']
        use_same_tasks_for_eval = kwargs['use_same_tasks_for_eval']
        if use_same_tasks_for_eval:
            assert num_eval_tasks == num_train_tasks
            env = MUJOCO_META_ENV_LIB[env_name](randomize_tasks=True, n_tasks=num_train_tasks) 
            train_tasks = list(range(num_train_tasks))
            eval_tasks = train_tasks
        else:
            env = MUJOCO_META_ENV_LIB[env_name](randomize_tasks=True,n_tasks=num_train_tasks+num_eval_tasks)
            train_tasks = list(range(num_train_tasks))
            eval_tasks = [i + num_train_tasks for i in range(num_eval_tasks)]
        env = OriNormalizedBoxEnv(env)
        return env, train_tasks, eval_tasks
    elif env_name in METAWORLD_SINGLE_ENVS:
        from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
                            ALL_V2_ENVIRONMENTS_GOAL_HIDDEN)
        goal_observable = kwargs["goal_observable"]
        if goal_observable:
            env_name += '-goal-observable'
            env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name]
        else:
            env_name += '-goal-hidden'
            env_cls = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN[env_name]
        env = env_cls(seed=seed)
        env =  MetaworldSingleWrapper(env)
        return env
    elif env_name in METAWORLD_META_ENVS:
        import metaworld
        if env_name == 'ML1':
            ml_env = metaworld.ML1(seed=seed)
        if env_name == 'ML10':
            ml_env = metaworld.ML10(seed=seed)
        elif env_name == 'ML45':
            ml_env = metaworld.ML45(seed=seed)
        for env_name, env_cls in ml_env.train_classes.items():
            env = env_cls()
            task = random.choice([task for task in ml_env.train_tasks  if task.env_name == env_name])

        return env, env.train_tasks, env.test_tasks

    elif env_name in MBPO_ENVS:
        from unstable_baselines.envs.mbpo import register_mbpo_environments
        register_mbpo_environments()
        env = gym.make(env_name, **kwargs)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        return env
    elif env_name in ATARI_ENVS:
        return wrap_atari_env(gym.make(env_name, render_mode="rgb_array"), **kwargs)
    elif env_name in DMC_ENVS:
        domain_name, task_name = env_name.split("-")
        env = DMCWrapper(domain_name, task_name, task_kwargs = {"random": seed}, **kwargs)
        return env
    else:
        print("Env {} not supported".format(env_name))
        exit(0)

class BaseEnvWrapper(gym.Wrapper):
    def __init__(self, env, **kwargs):
        super(BaseEnvWrapper, self).__init__(env)
        self.reward_scale = 1.0
        return


class ScaleRewardWrapper(BaseEnvWrapper):
    def __init__(self, env, **kwargs):
        super(ScaleRewardWrapper, self).__init__(env)
        self.reward_scale = kwargs['reward_scale']

    def step(self, action):
        try:
            s, r, d, info = self.env.step(action)
        except:
            print(action)
            assert 0
        scaled_reward = r * self.reward_scale
        return s, scaled_reward, d, info



# from https://github.com/joschu/modular_rl
# http://www.johndcook.com/blog/standard_deviation/

class RunningStat(object):
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)

    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape
class ZFilter:
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """

    def __init__(self, shape, demean=True, destd=True, clip=10.0):
        self.demean = demean
        self.destd = destd
        self.clip = clip

        self.rs = RunningStat(shape)

    def __call__(self, x, update=True):
        if update: self.rs.push(x)
        if self.demean:
            x = x - self.rs.mean
        if self.destd:
            x = x / (self.rs.std + 1e-8)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x

    def output_shape(self, input_space):
        return input_space.shape

class NormalizedBoxEnv(gym.Wrapper):
    """
    Normalize action to in [-1, 1].

    Optionally normalize observations and scale reward.
    """
    def __init__(
            self,
            env,
            normalize_obs=True,
            normalize_reward=False
    ):
        # self._wrapped_env needs to be called first because
        # Serializable.quick_init calls getattr, on this class. And the
        # implementation of getattr (see below) calls self._wrapped_env.
        # Without setting this first, the call to self._wrapped_env would call
        # getattr again (since it's not set yet) and therefore loop forever.
        super(NormalizedBoxEnv, self).__init__(env)
        self.normalize_obs = normalize_obs
        self.normalize_reward = normalize_reward
        self.running_obs , self.running_rew = None, None
        if self.normalize_obs:
            self.running_obs = ZFilter(env.observation_space.shape)
        if self.normalize_reward:
            self.running_rew = ZFilter((1,))

    def __getstate__(self):
        d = {}
        d["running_obs"] = deepcopy(self.running_obs)
        d["running_rew"] = deepcopy(self.running_rew)
        return d

    def __setstate__(self, d):
        self.running_obs = d["running_obs"]
        self.running_rew = d["running_rew"]

    def __copy__(self, env):
        self.__setstate__(env.__getstate__())

    def step(self, action, update=True):
        next_obs, reward, done, truncated, info  = self.env.step(action)
        if self.normalize_obs:
            next_obs = self.running_obs(next_obs)
        if self.normalize_reward:
            reward = self.running_rew(reward)
        return next_obs, reward, done, truncated, info


    def __getattr__(self, attrname):
        return getattr(self._wrapped_env, attrname)

    


class AtariWrapper(gym.Wrapper):
    def __init__(self, env, frameskip=4, resolution=(3, 105, 80), nstack=4, **kwargs):
        super(AtariWrapper, self).__init__(env)
        self.frameskip = frameskip
        self.nstack = nstack
        self.resolution = resolution
        self._obs_buffer = deque(maxlen=2)
        self.lives = 0
        self.was_real_done = True
        obs_space = Box(low=0, high=255, shape=self.resolution, dtype = np.uint8)

        low = np.repeat(obs_space.low, self.nstack, axis=0)
        high = np.repeat(obs_space.high, self.nstack, axis=0)

        self.stacked_obs = np.zeros(low.shape, low.dtype)
        self._observation_space = Box(low=low, high=high, shape = low.shape, dtype=np.uint8)

        self._action_space = env.action_space

    def step(self, action):
        total_reward = 0.0
        done = None
        combined_info = {}
        for _ in range(self.frameskip):
            obs, reward, done, info = self.env.step(action)
            obs = self.reshape_obs(obs)
            self._obs_buffer.append(obs)
            total_reward += reward
            combined_info.update(info)
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)

        self.stacked_obs = np.roll(self.stacked_obs, shift=-max_frame.shape[0], axis=0)
        if done:
            self.stacked_obs[...] = 0
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        self.stacked_obs[-max_frame.shape[0]:, ...] = max_frame
        return self.stacked_obs, total_reward, done, combined_info


    def reshape_obs(self, obs):
        if self.resolution[0] == 1:
            #rgb2gray
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)

            obs = cv2.resize(obs, (self.resolution[1],self.resolution[2]), interpolation=cv2.INTER_AREA)
            obs = obs[:, :, None]
        else:
            obs = np.array(Image.fromarray(obs).resize((self.resolution[2],self.resolution[1]),
                                                   resample=Image.Resampling.BILINEAR), dtype=np.uint8)
        obs = np.transpose(obs, [2, 0, 1])
        return obs

    def reset(self):
        info = {}
        if self.was_real_done:
            obs, step_info = self.env.reset()
            info.update(step_info)
            self._obs_buffer.clear()
            obs, step_info = self.env.reset()
            info.update(step_info)
            obs = self.reshape_obs(obs)
            self._obs_buffer.append(obs)
            self.stacked_obs[...] = 0
            self.stacked_obs[-obs.shape[0]:, ...] = obs
        else:
            obs, _, _, _, _ = self.env.step(0)
            obs = self.reshape_obs(obs)
            self.stacked_obs = np.roll(self.stacked_obs, shift=-obs.shape[0], axis=0)
            self.stacked_obs[-obs.shape[0]:, ...] = obs
        self.lives = self.env.unwrapped.ale.lives()
        return self.stacked_obs, info


cv2.ocl.setUseOpenCL(False)


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        info = {}
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            #noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)  # pylint: disable=E1101
            noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, truncated, step_info = self.env.step(self.noop_action)
            info.update(step_info)
            if done:
                obs, step_info = self.env.reset(**kwargs)
                info.update(step_info)
        return obs, info

    def step(self, ac):
        return self.env.step(ac)


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, truncated, info = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, truncated, info = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs, info

    def step(self, ac):
        return self.env.step(ac)


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, truncated, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _, info = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, frameskip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = frameskip

    def reset(self):
        return self.env.reset()

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, truncated, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, resolution):
        """Warp frames to 84x84 as done in the Nature paper and later work.
        Expects inputs to be of shape height x width x num_channels
        """
        gym.ObservationWrapper.__init__(self, env)
        self.width = resolution[0]
        self.height = resolution[1]
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(self.height, self.width, 1), dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]


class FrameStack(gym.Wrapper):
    def __init__(self, env, nstack):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        Expects inputs to be of shape num_channels x height x width.
        """
        gym.Wrapper.__init__(self, env)
        self.k = nstack
        self.frames = deque([], maxlen=nstack)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0] * nstack, shp[1], shp[2]), dtype=np.uint8)

    def reset(self):
        ob, info = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob(), info

    def step(self, action):
        ob, reward, done, truncated, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, truncated, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return np.array(LazyFrames(list(self.frames)), dtype=np.uint8)
    
    def render(self, **kwargs):
        return self.env.render()


class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers."""
        self._frames = frames

    def __array__(self, dtype=None):
        out = np.concatenate(self._frames, axis=0)
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._frames)

    def __getitem__(self, i):
        return self._frames[i]


class PyTorchFrame(gym.ObservationWrapper):
    """Image shape to num_channels x height x width"""

    def __init__(self, env):
        super(PyTorchFrame, self).__init__(env)
        shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(shape[-1], shape[0], shape[1]), dtype=np.uint8)

    def observation(self, observation):
        return np.rollaxis(observation, 2)



def wrap_atari_env(env, resolution, noop_max, frameskip, nstack, **kwargs):
    env = NoopResetEnv(env, noop_max=noop_max)
    env = MaxAndSkipEnv(env, frameskip=frameskip)
    env = EpisodicLifeEnv(env)
    if  env.unwrapped.get_action_meanings()[1] == 'FIRE' and len(env.unwrapped.get_action_meanings()) >= 3:
        env = FireResetEnv(env)
    env = WarpFrame(env, resolution=resolution)
    env = PyTorchFrame(env)
    #env = ClipRewardEnv(env)
    env = FrameStack(env, nstack)
    return env


import inspect
import sys


class Serializable(object):

    def __init__(self, *args, **kwargs):
        self.__args = args
        self.__kwargs = kwargs

    def quick_init(self, locals_):
        if getattr(self, "_serializable_initialized", False):
            return
        if sys.version_info >= (3, 0):
            spec = inspect.getfullargspec(self.__init__)
            # Exclude the first "self" parameter
            if spec.varkw:
                kwargs = locals_[spec.varkw].copy()
            else:
                kwargs = dict()
            if spec.kwonlyargs:
                for key in spec.kwonlyargs:
                    kwargs[key] = locals_[key]
        else:
            spec = inspect.getargspec(self.__init__)
            if spec.keywords:
                kwargs = locals_[spec.keywords]
            else:
                kwargs = dict()
        if spec.varargs:
            varargs = locals_[spec.varargs]
        else:
            varargs = tuple()
        in_order_args = [locals_[arg] for arg in spec.args][1:]
        self.__args = tuple(in_order_args) + varargs
        self.__kwargs = kwargs
        setattr(self, "_serializable_initialized", True)

    def __getstate__(self):
        return {"__args": self.__args, "__kwargs": self.__kwargs}

    def __setstate__(self, d):
        # convert all __args to keyword-based arguments
        if sys.version_info >= (3, 0):
            spec = inspect.getfullargspec(self.__init__)
        else:
            spec = inspect.getargspec(self.__init__)
        in_order_args = spec.args[1:]
        out = type(self)(**dict(zip(in_order_args, d["__args"]), **d["__kwargs"]))
        self.__dict__.update(out.__dict__)

    @classmethod
    def clone(cls, obj, **kwargs):
        assert isinstance(obj, Serializable)
        d = obj.__getstate__()
        d["__kwargs"] = dict(d["__kwargs"], **kwargs)
        out = type(obj).__new__(type(obj))
        out.__setstate__(d)
        return out


class ProxyEnv(Serializable, gym.Env):
    def __init__(self, wrapped_env):
        Serializable.quick_init(self, locals())
        self._wrapped_env = wrapped_env
        self.action_space = self._wrapped_env.action_space
        self.observation_space = self._wrapped_env.observation_space

    @property
    def wrapped_env(self):
        return self._wrapped_env

    def reset(self, **kwargs):
        return self._wrapped_env.reset(**kwargs)

    def step(self, action):
        return self._wrapped_env.step(action)

    def render(self, *args, **kwargs):
        return self._wrapped_env.render(*args, **kwargs)

    def log_diagnostics(self, paths, *args, **kwargs):
        if hasattr(self._wrapped_env, 'log_diagnostics'):
            self._wrapped_env.log_diagnostics(paths, *args, **kwargs)

    @property
    def horizon(self):
        return self._wrapped_env.horizon

    def terminate(self):
        if hasattr(self.wrapped_env, "terminate"):
            self.wrapped_env.terminate()


class OriNormalizedBoxEnv(ProxyEnv, Serializable):
    """
    Normalize action to in [-1, 1].

    Optionally normalize observations and scale reward.
    """
    def __init__(
            self,
            env,
            reward_scale=1.,
            obs_mean=None,
            obs_std=None,
    ):
        # self._wrapped_env needs to be called first because
        # Serializable.quick_init calls getattr, on this class. And the
        # implementation of getattr (see below) calls self._wrapped_env.
        # Without setting this first, the call to self._wrapped_env would call
        # getattr again (since it's not set yet) and therefore loop forever.
        self._wrapped_env = env
        # Or else serialization gets delegated to the wrapped_env. Serialize
        # this env separately from the wrapped_env.
        self._serializable_initialized = False
        Serializable.quick_init(self, locals())
        ProxyEnv.__init__(self, env)
        self._should_normalize = not (obs_mean is None and obs_std is None)
        if self._should_normalize:
            if obs_mean is None:
                obs_mean = np.zeros_like(env.observation_space.low)
            else:
                obs_mean = np.array(obs_mean)
            if obs_std is None:
                obs_std = np.ones_like(env.observation_space.low)
            else:
                obs_std = np.array(obs_std)
        self._reward_scale = reward_scale
        self._obs_mean = obs_mean
        self._obs_std = obs_std
        ub = np.ones(self._wrapped_env.action_space.shape)
        self.action_space = gym.spaces.Box(-1 * ub, ub)

    def estimate_obs_stats(self, obs_batch, override_values=False):
        if self._obs_mean is not None and not override_values:
            raise Exception("Observation mean and std already set. To "
                            "override, set override_values to True.")
        self._obs_mean = np.mean(obs_batch, axis=0)
        self._obs_std = np.std(obs_batch, axis=0)

    def _apply_normalize_obs(self, obs):
        return (obs - self._obs_mean) / (self._obs_std + 1e-8)

    def __getstate__(self):
        d = Serializable.__getstate__(self)
        # Add these explicitly in case they were modified
        d["_obs_mean"] = self._obs_mean
        d["_obs_std"] = self._obs_std
        d["_reward_scale"] = self._reward_scale
        return d

    def __setstate__(self, d):
        Serializable.__setstate__(self, d)
        self._obs_mean = d["_obs_mean"]
        self._obs_std = d["_obs_std"]
        self._reward_scale = d["_reward_scale"]

    def step(self, action):
        lb = self._wrapped_env.action_space.low
        ub = self._wrapped_env.action_space.high
        scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)
        scaled_action = np.clip(scaled_action, lb, ub)

        wrapped_step = self._wrapped_env.step(scaled_action)
        next_obs, reward, done, truncated, info = wrapped_step
        if self._should_normalize:
            next_obs = self._apply_normalize_obs(next_obs)
        return next_obs, reward * self._reward_scale, done, truncated, info

    def __str__(self):
        return "Normalized: %s" % self._wrapped_env

    def log_diagnostics(self, paths, **kwargs):
        if hasattr(self._wrapped_env, "log_diagnostics"):
            return self._wrapped_env.log_diagnostics(paths, **kwargs)
        else:
            return None

    def __getattr__(self, attrname):
        return getattr(self._wrapped_env, attrname)

class MetaworldSingleWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.__dict__.update(env.__dict__)

    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        truncated = False
        return next_obs, reward, done, truncated, info

    def reset(self):
        obs = self.env.reset()
        info = {}
        return obs, info
    
class MetaWorldMetaWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.__dict__.update(env.__dict__)

    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        truncated = False
        return next_obs, reward, done, truncated, info

    def reset(self):
        obs = self.env.reset()
        info = {}
        return obs, info
    
def _spec_to_box(spec, dtype):
    def extract_min_max(s):
        assert s.dtype == np.float64 or s.dtype == np.float32
        dim = np.int(np.prod(s.shape))
        if type(s) == specs.Array:
            bound = np.inf * np.ones(dim, dtype=np.float32)
            return -bound, bound
        elif type(s) == specs.BoundedArray:
            zeros = np.zeros(dim, dtype=np.float32)
            return s.minimum + zeros, s.maximum + zeros

    mins, maxs = [], []
    for s in spec:
        mn, mx = extract_min_max(s)
        mins.append(mn)
        maxs.append(mx)
    low = np.concatenate(mins, axis=0).astype(dtype)
    high = np.concatenate(maxs, axis=0).astype(dtype)
    assert low.shape == high.shape
    return spaces.Box(low, high, dtype=dtype)


def _flatten_obs(obs):
    obs_pieces = []
    for v in obs.values():
        flat = np.array([v]) if np.isscalar(v) else v.ravel()
        obs_pieces.append(flat)
    return np.concatenate(obs_pieces, axis=0)

class DMCWrapper(core.Env):
    def __init__(
        self,
        domain_name,
        task_name,
        task_kwargs=None,
        visualize_reward=True,
        from_pixels=False,
        height=84,
        width=84,
        camera_id=0,
        frame_skip=1,
        environment_kwargs=None,
        channels_first=True
    ):
        assert 'random' in task_kwargs, 'please specify a seed, for deterministic behaviour'
        self._from_pixels = from_pixels
        self._height = height
        self._width = width
        self._camera_id = camera_id
        self._frame_skip = frame_skip
        self._channels_first = channels_first

        # create task
        from dm_control import suite
        self._env = suite.load(
            domain_name=domain_name,
            task_name=task_name,
            task_kwargs=task_kwargs,
            visualize_reward=visualize_reward,
            environment_kwargs=environment_kwargs
        )

        # true and normalized action spaces
        self._true_action_space = _spec_to_box([self._env.action_spec()], np.float32)
        self._norm_action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=self._true_action_space.shape,
            dtype=np.float32
        )

        # create observation space
        if from_pixels:
            shape = [3, height, width] if channels_first else [height, width, 3]
            self._observation_space = spaces.Box(
                low=0, high=255, shape=shape, dtype=np.uint8
            )
        else:
            self._observation_space = _spec_to_box(
                self._env.observation_spec().values(),
                np.float64
            )
            
        self._state_space = _spec_to_box(
            self._env.observation_spec().values(),
            np.float64
        )
        
        self.current_state = None

        # set seed
        self.seed(seed=task_kwargs.get('random', 1))

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _get_obs(self, time_step):
        if self._from_pixels:
            obs = self.render(
                height=self._height,
                width=self._width,
                camera_id=self._camera_id
            )
            if self._channels_first:
                obs = obs.transpose(2, 0, 1).copy()
        else:
            obs = _flatten_obs(time_step.observation)
        return obs

    def _convert_action(self, action):
        action = action.astype(np.float64)
        true_delta = self._true_action_space.high - self._true_action_space.low
        norm_delta = self._norm_action_space.high - self._norm_action_space.low
        action = (action - self._norm_action_space.low) / norm_delta
        action = action * true_delta + self._true_action_space.low
        action = action.astype(np.float32)
        return action

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def state_space(self):
        return self._state_space

    @property
    def action_space(self):
        return self._norm_action_space

    @property
    def reward_range(self):
        return 0, self._frame_skip

    def seed(self, seed):
        self._true_action_space.seed(seed)
        self._norm_action_space.seed(seed)
        self._observation_space.seed(seed)

    def step(self, action):
        assert self._norm_action_space.contains(action)
        action = self._convert_action(action)
        assert self._true_action_space.contains(action)
        reward = 0
        extra = {'internal_state': self._env.physics.get_state().copy()}

        for _ in range(self._frame_skip):
            time_step = self._env.step(action)
            reward += time_step.reward or 0
            done = time_step.last()
            if done:
                break
        obs = self._get_obs(time_step)
        self.current_state = _flatten_obs(time_step.observation)
        extra['discount'] = time_step.discount
        return obs, reward, done, False, extra

    def reset(self):
        time_step = self._env.reset()
        self.current_state = _flatten_obs(time_step.observation)
        obs = self._get_obs(time_step)
        return obs, {}

    def render(self, mode='rgb_array', height=None, width=None, camera_id=0):
        assert mode == 'rgb_array', 'only support rgb_array mode, given %s' % mode
        height = height or self._height
        width = width or self._width
        camera_id = camera_id or self._camera_id
        return self._env.physics.render(
            height=height, width=width, camera_id=camera_id
        )