import pytest
from tqdm import tqdm

from unstable_baselines.common.agents import RandomAgent
from unstable_baselines.common.rollout import RolloutBuffer


buffer_size = 1000
gamma = 0.99
advantage_type = "gae"
normalize = True
gae_lambda = 0.95
max_trajectory_length = 1000


# Two environments for fixed length episode and not fixed
@pytest.fixture(params=['HalfCheetah-v3', 'Hopper-v3'])
def _init_rollout_buffer(request, scope='function'):
    env_name = request.param
    env = gym.make(env_name)
    state_space = env.observation_space
    action_space = env.action_space

    agent = RandomAgent(state_space, action_space)
    rollout_buffer = RolloutBuffer(
        state_space,
        action_space,
        size=buffer_size,
        gamma=gamma,
        advantage_type=advantage_type,
        normalize_advantage=normalize,
        gae_lambda=gae_lambda,
        mex_ep_length=max_trajectory_length
    )
    yield rollout_buffer, env
    # it will return `rollout_buffer` and `env` in the test function
    # and you can add other process after the test function here


class TestRolloutBuffer:

    def test_discount_sum(self, _init_rollout_buffer):
        buffer, env = _init_rollout_buffer
        state = env.reset()
        value = 2
        last_value = 0
        action = env.action_space.sample()
        for steps in range(1000):
            buffer.store(state, action, steps, value, 1)
        buffer.finish_path(last_value=last_value)

        assert buffer.return_buffer[-2] == pytest.approx(998 + gamma * 999)
        assert buffer.advantage_buffer[-2] == pytest.approx(998+gamma*value-value + gae_lambda*gamma*(999+gamma*last_value-value))
