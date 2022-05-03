import gym
#from mujoco_py import GlfwContext
#GlfwContext(offscreen=True) 
env = gym.make("Hopper-v3")
env.reset()
#env.render()
img = env.render(mode="rgb_array")
