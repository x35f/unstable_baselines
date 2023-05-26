import numpy as np
import pytest
import os
import logging
logger = logging.getLogger(__name__)
from unstable_baselines.envs.mujoco_meta import gym
from unstable_baselines.envs.mujoco_meta.gym import envs
from unstable_baselines.envs.mujoco_meta.gym.envs.tests.spec_list import spec_list


# This runs a smoketest on each official registered env. We may want
# to try also running environments which are not officially registered
# envs.
@pytest.mark.parametrize("spec", spec_list)
def test_env(spec):
    env = spec.make()
    ob_space = env.observation_space
    act_space = env.action_space
    ob = env.reset()
    assert ob_space.contains(ob), 'Reset observation: {!r} not in space'.format(ob)
    a = act_space.sample()
    observation, reward, done, _info = env.step(a)
    assert ob_space.contains(observation), 'Step observation: {!r} not in space'.format(observation)
    assert np.isscalar(reward), "{} is not a scalar for {}".format(reward, env)
    assert isinstance(done, bool), "Expected {} to be a boolean".format(done)

    for mode in env.metadata.get('render.modes', []):
        env.render(mode=mode)
    env.render(close=True)

    # Make sure we can render the environment after close.
    for mode in env.metadata.get('render.modes', []):
        env.render(mode=mode)
    env.render(close=True)

    env.close()

# Run a longer rollout on some environments
def test_random_rollout():
    for env in [envs.make('CartPole-v0'), envs.make('FrozenLake-v0')]:
        agent = lambda ob: env.action_space.sample()
        ob = env.reset()
        for _ in range(10):
            assert env.observation_space.contains(ob)
            a = agent(ob)
            assert env.action_space.contains(a)
            (ob, _reward, done, _info) = env.step(a)
            if done: break

def test_double_close():
    class TestEnv(gym.Env):
        def __init__(self):
            self.close_count = 0

        def _close(self):
            self.close_count += 1

    env = TestEnv()
    assert env.close_count == 0
    env.close()
    assert env.close_count == 1
    env.close()
    assert env.close_count == 1
