from unstable_baselines.envs.mujoco_meta.gym.envs.mujoco.mujoco_env import MujocoEnv
# ^^^^^ so that user gets the correct error
# message if mujoco is not installed correctly
from unstable_baselines.envs.mujoco_meta.gym.envs.mujoco.ant import AntEnv
from unstable_baselines.envs.mujoco_meta.gym.envs.mujoco.half_cheetah import HalfCheetahEnv
from unstable_baselines.envs.mujoco_meta.gym.envs.mujoco.hopper import HopperEnv
from unstable_baselines.envs.mujoco_meta.gym.envs.mujoco.walker2d import Walker2dEnv
from unstable_baselines.envs.mujoco_meta.gym.envs.mujoco.humanoid import HumanoidEnv
from unstable_baselines.envs.mujoco_meta.gym.envs.mujoco.inverted_pendulum import InvertedPendulumEnv
from unstable_baselines.envs.mujoco_meta.gym.envs.mujoco.inverted_double_pendulum import InvertedDoublePendulumEnv
from unstable_baselines.envs.mujoco_meta.gym.envs.mujoco.reacher import ReacherEnv
from unstable_baselines.envs.mujoco_meta.gym.envs.mujoco.swimmer import SwimmerEnv
from unstable_baselines.envs.mujoco_meta.gym.envs.mujoco.humanoidstandup import HumanoidStandupEnv
