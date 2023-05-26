# -*- coding: utf-8 -*-
from unstable_baselines.envs.mujoco_meta.gym import error, envs
from unstable_baselines.envs.mujoco_meta.gym.envs import registration
from unstable_baselines.envs.mujoco_meta.gym.envs.classic_control import cartpole

def test_make():
    env = envs.make('CartPole-v0')
    assert env.spec.id == 'CartPole-v0'
    assert isinstance(env.unwrapped, cartpole.CartPoleEnv)

def test_make_deprecated():
    try:
        envs.make('Humanoid-v0')
    except error.Error:
        pass
    else:
        assert False

def test_spec():
    spec = envs.spec('CartPole-v0')
    assert spec.id == 'CartPole-v0'

def test_missing_lookup():
    registry = registration.EnvRegistry()
    registry.register(id='Test-v0', entry_point=None)
    registry.register(id='Test-v15', entry_point=None)
    registry.register(id='Test-v9', entry_point=None)
    registry.register(id='Other-v100', entry_point=None)
    try:
        registry.spec('Test-v1')  # must match an env name but not the version above
    except error.DeprecatedEnv:
        pass
    else:
        assert False

    try:
        registry.spec('Unknown-v1')
    except error.UnregisteredEnv:
        pass
    else:
        assert False

def test_malformed_lookup():
    registry = registration.EnvRegistry()
    try:
        registry.spec(u'“Breakout-v0”')
    except error.Error as e:
        assert 'malformed environment ID' in '{}'.format(e), 'Unexpected message: {}'.format(e)
    else:
        assert False
