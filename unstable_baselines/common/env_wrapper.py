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

from gym import spaces

MUJOCO_SINGLE_ENVS = [
    'Ant-v2', 'Ant-v3',
    'HalfCheetah-v2', 'HalfCheetah-v3',
    'Hopper-v2', 'Hopper-v3',
    'Humanoid-v2', 'Humanoid-v3',
    'InvertedDoublePendulum-v2',
    'InvertedPendulum-v2', 'InvertedPendulum-v3',
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
ATARI_ENVS = [
    'Pong-v4', 'PongNoFrameskip-v4', 'QbertNoFrameskip-v4', 
    'BreakoutNoFrameskip-v4',"MsPacmanNoFrameskip-v4",
    "SeaquestNoFrameskip-v4",'Qbert-v4', 
    'Breakout-v4',"MsPacman-v4",
    "Seaquest-v4", "BeamRider-v4",
    "DemonAttack-v4", "SpaceInvaders-v4",
    "TimePilot-v4",
    
]

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
    elif env_name in ATARI_ENVS:
        return gym.make(env_name, **kwargs)
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
        if self.was_real_done:
            obs = self.env.reset()
            self._obs_buffer.clear()
            obs = self.env.reset()
            obs = self.reshape_obs(obs)
            self._obs_buffer.append(obs)
            self.stacked_obs[...] = 0
            self.stacked_obs[-obs.shape[0]:, ...] = obs
        else:
            obs, _, _, _ = self.env.step(0)
            obs = self.reshape_obs(obs)
            self.stacked_obs = np.roll(self.stacked_obs, shift=-obs.shape[0], axis=0)
            self.stacked_obs[-obs.shape[0]:, ...] = obs
        self.lives = self.env.unwrapped.ale.lives()
        return self.stacked_obs


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
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            #noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)  # pylint: disable=E1101
            noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

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
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

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
        obs, reward, done, info = self.env.step(action)
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
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


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
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

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
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return np.array(LazyFrames(list(self.frames)), dtype=np.uint8)


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
