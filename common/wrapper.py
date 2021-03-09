import gym
from abc import abstractmethod

class BaseEnvWrapper(gym.Wrapper):
    def __init__(self, env):
        super(BaseEnvWrapper, self).__init__(env)
        pass

    @abstractmethod
    def step(action):
        pass

    @abstractmethod
    def reset(self):
        pass



# class MonitorWrapper(BaseEnvWrapper):
#     def __init__(self, env, log_path, log_interval):
#         super(MonitorWrapper, self).__init__(env)