from unstable_baselines.envs.mujoco_meta import gym
from unstable_baselines.envs.mujoco_meta.gym.spaces import prng
import numpy as np

class MultiBinary(gym.Space):
    def __init__(self, n):
        self.n = n
    def sample(self):
        return prng.np_random.randint(low=0, high=2, size=self.n)
    def contains(self, x):
        return ((x==0) | (x==1)).all()
    def to_jsonable(self, sample_n):
        return sample_n.tolist()
    def from_jsonable(self, sample_n):
        return np.array(sample_n)