from common.wrapper import BaseEnvWrapper

class PPOWrapper(BaseEnvWrapper):
    def __init__(self, env, reward_scale=1,**kwargs):
        super(PPOWrapper, self).__init__(env)
        self.env = env
        self.reward_scale = reward_scale

    def step(self, action):
        s, r, d, info = self.env.step(action)
        scaled_reward = r * self.reward_scale
        return s, scaled_reward, d, info