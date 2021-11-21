from envs.mujoco_meta.gym.envs.mujoco.mujoco_env import MujocoEnv
# ^^^^^ so that user gets the correct error
# message if mujoco is not installed correctly
from envs.mujoco_meta.gym.envs.mujoco.ant import AntEnv
from envs.mujoco_meta.gym.envs.mujoco.half_cheetah import HalfCheetahEnv
from envs.mujoco_meta.gym.envs.mujoco.hopper import HopperEnv
from envs.mujoco_meta.gym.envs.mujoco.walker2d import Walker2dEnv
from envs.mujoco_meta.gym.envs.mujoco.humanoid import HumanoidEnv
from envs.mujoco_meta.gym.envs.mujoco.inverted_pendulum import InvertedPendulumEnv
from envs.mujoco_meta.gym.envs.mujoco.inverted_double_pendulum import InvertedDoublePendulumEnv
from envs.mujoco_meta.gym.envs.mujoco.reacher import ReacherEnv
from envs.mujoco_meta.gym.envs.mujoco.swimmer import SwimmerEnv
from envs.mujoco_meta.gym.envs.mujoco.humanoidstandup import HumanoidStandupEnv
