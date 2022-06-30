import gym
import numpy as np
from copy import deepcopy
# from mujoco_py import GlfwContext
# GlfwContext(offscreen=True) 


MUJOCO_SINGLE_ENVS = [
    'Ant-v2', 'Ant-v3',
    'HalfCheetah-v2', 'HalfCheetah-v3',
    'Hopper-v2', 'Hopper-v3',
    'Humanoid-v2', 'Humanoid-v3',
    'InvertedDoublePendulum-v2',
    'InvertedPendulum-v2',
    'Swimmer-v2', 'Swimmer-v3',
    'Walker2d-v2', 'Walker2d-v3',
    'Pusher-v2',
    'Reacher-v2',
    'Striker-v2',
    'Thrower-v2',
    'CartPole-v1',
    'MountainCar-v0'
    ]

MUJOCO_META_ENVS = [
    'point-robot', 'sparse-point-robot', 'walker-rand-params', 
    'humanoid-dir', 'hopper-rand-params', 'ant-dir', 
    'cheetah-vel', 'cheetah-dir', 'ant-goal']

METAWORLD_ENVS = ['MetaWorld']

MBPO_ENVS = [
    'AntTruncatedObs-v2',
    'HumanoidTruncatedObs-v2',
    ]
ATARI_ENVS = ['']


def get_env(env_name, **kwargs):
    if env_name in MUJOCO_SINGLE_ENVS:
        return gym.make(env_name, **kwargs)
    elif env_name in MUJOCO_META_ENVS:
        from unstable_baselines.envs.mujoco_meta.rlkit_envs import ENVS as MUJOCO_META_ENV_LIB
        return MUJOCO_META_ENV_LIB[env_name](**kwargs)
    elif env_name in METAWORLD_ENVS:
        raise NotImplementedError
    elif env_name in MBPO_ENVS:
        from unstable_baselines.envs.mbpo import register_mbpo_environments
        register_mbpo_environments()
        env = gym.make(env_name)
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

    def step(self, action, update=True):
        next_obs, reward, done, info  = self.env.step(action)
        if self.normalize_obs:
            next_obs = self.running_obs(next_obs)
        if self.normalize_reward:
            reward = self.running_rew(reward)
        return next_obs, reward, done, info


    def __getattr__(self, attrname):
        return getattr(self._wrapped_env, attrname)
