import os
import gym
import mujoco_py
import matplotlib as plt
import click
from util import LOGGER, load_config
from models import SACAgent
from util import set_device
from common.buffer import REPLAY_BUFFER
import json
@click.command()
@click.argument("config-path",type=str)
@click.option("--log-dir", default = "log")
@click.option("--gpu-id", type = int, default = -1)
def main(config_path, log_dir, gpu_id, **kwargs):
    #todo: add load and update parameters function
    json_file = os.path.join("configs","{}.json".format(config_path))
    assert os.path.exists(json_file)
    args = load_config(json_file, kwargs)
    #initialize device
    set_device(gpu_id)
    #initialize logger
    logger = LOGGER(log_dir, prefix = env_name)
    logger.log_str("logging to {}".format(log_dir))
    #initialize environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    #initialize buffer
    max_buffer_size = args['max_buffer_size']
    buffer = REPLAY_BUFFER(state_dim, action_dim, max_buffer_size)
    #initialize agent
    agent = SACAgent(env.observation_space, env.action_space, args['agent_parameters'])
    #initialize model
    trainer  = SACAgent(
        agent,
        env,
        buffer,
        logger,
        **kwargs
    )
    trainer.train(args['max_iteration'])




if __name__ == "__main__":
    main()