import gym
import numpy as np
from copy import deepcopy
from collections import deque
# from mujoco_py import GlfwContext
# GlfwContext(offscreen=True) 
from gym.envs.registration import register
from gym.spaces.box import Box
from PIL import Image

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

PYBULLET_ENVS = ['takeoff-aviary-v0', 'hover-aviary-v0', 'flythrugate-aviary-v0', 'tune-aviary-v0']

SAFE_ENVS = ['Safexp-PointGoal1-v0', "DoggoGoal-v0", "DoggoPush-v0", "CarGoal-v0", "CarPush-v0"]


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
        env = gym.make(env_name, **kwargs)
        return env
    elif env_name in PYBULLET_ENVS:
        from gym_pybullet_drones.envs.single_agent_rl.TakeoffAviary import TakeoffAviary
        from gym_pybullet_drones.envs.single_agent_rl.HoverAviary import HoverAviary
        from gym_pybullet_drones.envs.single_agent_rl.FlyThruGateAviary import FlyThruGateAviary
        from gym_pybullet_drones.envs.single_agent_rl.TuneAviary import TuneAviary
        from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
        obs = ObservationType("kin")
        act =  ActionType('dyn')
        AGGR_PHY_STEPS = 5
        if env_name == "takeoff-aviary-v0":
            return TakeoffAviary(obs=obs, act=act, aggregate_phy_steps=AGGR_PHY_STEPS)
        elif env_name == "hover-aviary-v0":
            return HoverAviary(obs=obs, act=act, aggregate_phy_steps=AGGR_PHY_STEPS)
        elif env_name == "flythrugate-aviary-v0":
            return FlyThruGateAviary(obs=obs, act=act, aggregate_phy_steps=AGGR_PHY_STEPS)
        elif env_name == "tune-aviary-v0":
            act =  ActionType('tun')
            return TuneAviary(obs=obs, act=act, aggregate_phy_steps=AGGR_PHY_STEPS)
    elif env_name in SAFE_ENVS:
        return gym.make(env_name)
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


class AtariWrapper(gym.Wrapper):
    def __init__(self, env, frameskip=4, resolution=(105, 80, 3), nstack=4):
        super(AtariWrapper, self).__init__(env)
        self.frameskip = frameskip
        self.nstack = nstack
        self.resolution = resolution
        self._obs_buffer = deque(maxlen=2)

        obs_space = Box(low=0, high=255, shape=self.res, dtype = np.uint8)

        low = np.repeat(obs_space.low, self.nstack, axis=-1)
        high = np.repeat(obs_space.high, self.nstack, axis=-1)

        self.stacked_obs = np.zeros(low.shape, low.dtype)
        self._observation_space = Box(low=low, high=high, shape = low.shape, dtype=np.uint8)

        self._action_space = env.action_space

    def step(self, action):
        total_reward = 0.0
        done = None
        combined_info = {}
        for _ in range(self.frameskip):
            obs, reward, done, info = self.env.step(action[0])
            obs = self.reshape_obs(obs)
            self._obs_buffer.append(obs)
            total_reward += reward
            combined_info.update(info)
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)

        self.stacked_obs = np.roll(self.stacked_obs, shift=-1, axis=-1)
        if done:
            self.stacked_obs[...] = 0
        self.stacked_obs[..., -max_frame.shape[-1]:] = max_frame
        return self.stacked_obs, total_reward, done, combined_info


    def reshape_obs(self, obs):
        obs = np.array(Image.fromarray(obs[0]).resize((self.res[0],self.res[1]),
                                                   resample=Image.BILINEAR), dtype=np.uint8)
        return obs.reshape(self.res)

    def reset(self):
        self._obs_buffer.clear()
        obs = self.env.reset()
        obs = self.reshape_obs(obs)
        self._obs_buffer.append(obs)
        return obs