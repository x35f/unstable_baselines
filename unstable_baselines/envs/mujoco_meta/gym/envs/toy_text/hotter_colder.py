from unstable_baselines.envs.mujoco_meta import gym
from unstable_baselines.envs.mujoco_meta.gym import spaces
from unstable_baselines.envs.mujoco_meta.gym.utils import seeding
import numpy as np


class HotterColder(gym.Env):
    """Hotter Colder
    The goal of hotter colder is to guess closer to a randomly selected number

    After each step the agent receives an observation of:
    0 - No guess yet submitted (only after reset)
    1 - Guess is lower than the target
    2 - Guess is equal to the target
    3 - Guess is higher than the target

    The rewards is calculated as:
    (min(action, self.number) + self.range) / (max(action, self.number) + self.range)

    Ideally an agent will be able to recognise the 'scent' of a higher reward and
    increase the rate in which is guesses in that direction until the reward reaches
    its maximum
    """
    def __init__(self):
        self.range = 1000  # +/- value the randomly select number can be between
        self.bounds = 2000  # Action space bounds

        self.action_space = spaces.Box(low=np.array([-self.bounds]), high=np.array([self.bounds]))
        self.observation_space = spaces.Discrete(4)

        self.number = 0
        self.guess_count = 0
        self.guess_max = 200
        self.observation = 0

        self._seed()
        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action)

        if action < self.number:
            self.observation = 1

        elif action == self.number:
            self.observation = 2

        elif action > self.number:
            self.observation = 3

        reward = ((min(action, self.number) + self.bounds) / (max(action, self.number) + self.bounds)) ** 2

        self.guess_count += 1
        done = self.guess_count >= self.guess_max

        return self.observation, reward[0], done, {"number": self.number, "guesses": self.guess_count}

    def _reset(self):
        self.number = self.np_random.uniform(-self.range, self.range)
        self.guess_count = 0
        self.observation = 0
        return self.observation
