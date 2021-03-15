from common.wrapper import BaseEnvWrapper

class REDQWrapper(BaseEnvWrapper):
    def __init__(self, env, **kwargs):
        super(REDQWrapper, self).__init__(env)
        self.env = env
        self.reward_scale = kwargs['reward_scale']

    def step(self, action):
        s, r, d, info = self.env.step(action)
        scaled_reward = r * self.reward_scale
        return s, scaled_reward, d, info