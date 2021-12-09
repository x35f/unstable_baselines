import pytest


baselines = ['ddpg', 'dqn', 'ppo', 'redq', 'sac', 'tdn_sac']


def pytest_addoption(parser):
    parser.addoption("--algo", type=str, default='all', 
        help="Test option: [all, ddpg, dqn, ppo, redq, sac, tdn_sac]")


@pytest.fixture(scope='session', autouse=True)
def algo(request):
    result = request.config.getoption("--algo")
    print(result)
    return result
