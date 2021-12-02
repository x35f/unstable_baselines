from unstable_baselines.envs.mujoco_meta.base import MetaEnv
from unstable_baselines.envs.mujoco_meta.gym.envs.registration import register

register(
    id='Walker2DRandParams-v0',
    entry_point='envs.mujoco_meta.walker2d_rand_params:Walker2DRandParamsEnv',
)

register(
    id='HopperRandParams-v0',
    entry_point='envs.mujoco_meta.hopper_rand_params:HopperRandParamsEnv',
)

register(
    id='PR2Env-v0',
    entry_point='envs.mujoco_meta.pr2_env_reach:PR2Env',
)


