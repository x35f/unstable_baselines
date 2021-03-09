import gym
from abc import abstractmethod

class BaseEnvWrapper(gym.Wrapper):
    def __init__(self, env):
        super(BaseEnvWrapper, self).__init__(env)
        pass
