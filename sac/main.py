import os
import gym
import mujoco_py
import matplotlib as plt
import click
from util import LOGGER, load_config
from models import SACAgent
from util import set_device
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
    #initialize model
    model  = SACAgent(
        logger,
        env,
        args,
        **kwargs
    )
    model.train()




if __name__ == "__main__":
    main()