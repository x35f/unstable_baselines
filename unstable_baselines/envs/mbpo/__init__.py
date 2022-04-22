import gym

# from .AntTruncated import AntTruncatedObsEnv
# from .HumanoidTruncated import HumanoidTruncatedObsEnv
# mbpo_env_dict = {
#     "AntTruncated-v2": AntTruncated,
#     "HumanoidTruncatedObs":HumanoidTruncatedObsEnv
# }

MBPO_ENVIRONMENT_SPECS = (
	{
        'id': 'AntTruncatedObs-v2',
        'entry_point': ('unstable_baselines.envs.mbpo.AntTruncated:AntTruncatedObsEnv'),
    },
	{
        'id': 'HumanoidTruncatedObs-v2',
        'entry_point': ('unstable_baselines.envs.mbpo.HumanoidTruncated:HumanoidTruncatedObsEnv'),
    },
)

def register_mbpo_environments():
    for mbpo_environment in MBPO_ENVIRONMENT_SPECS:
        gym.register(**mbpo_environment)

    gym_ids = tuple(
        environment_spec['id']
        for environment_spec in  MBPO_ENVIRONMENT_SPECS)

    return gym_ids