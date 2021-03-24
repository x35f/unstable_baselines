from common.wrapper import BaseEnvWrapper

class SACWrapper(BaseEnvWrapper):
    def __init__(self, env, **kwargs):
        super(SACWrapper, self).__init__(env)
        self.env = env
        self.reward_scale = kwargs['reward_scale']

    def step(self, action):
        try:
            s, r, d, info = self.env.step(action)
        except e:
            print(e)
            print(action)
        scaled_reward = r * self.reward_scale
        return s, scaled_reward, d, info