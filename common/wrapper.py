import gym
from abc import abstractmethod

class BaseEnvWrapper(gym.Wrapper):
    def __init__(self, env, **kwargs):
        super(BaseEnvWrapper, self).__init__(env)
        pass


class ScaleRewardWrapper(BaseEnvWrapper):
    def __init__(self, env, **kwargs):
        super(ScaleRewardWrapper, self).__init__(env)
        self.env = env
        self.reward_scale = kwargs['reward_scale']

    def step(self, action):
        try:
            s, r, d, info = self.env.step(action)
        except:
            print(action)
            assert 0
        scaled_reward = r * self.reward_scale
        return s, scaled_reward, d, info