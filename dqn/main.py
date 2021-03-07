import os
import gym
import mujoco_py
import matplotlib as plt
import click
from util import LOGGER, update_parameters
from models import DQNAgent
from util import set_device
import json
@click.command()
@click.argument("env-name",type=str)
@click.option("--log-dir", default = "log")
@click.option("--gpu", type = int, default = -1)
@click.option("--max-buffer-size", type = int, default = 10000)
@click.option("--max-epoch", type = int, default = 100000)
@click.option("--num-sample-steps-per-epoch", type = int, default = 500)
@click.option("--num-update-steps-per-epoch", type = int, default = 10)
@click.option("--max-traj-length", type = int, default = 500)
@click.option("--epsilon", type = float, default = 0.2)
@click.option("--learning-rate", type = float, default = 0.0005)
@click.option("--update-q-target-inverval", type = int, default = 100)
@click.option("--batch-size", type = int, default = 64)
@click.option("--gamma", type = float, default = 0.95)
@click.option("--tau", type = float, default = 0.99)
@click.option("--num-test-trajs", type = int, default = 3)
@click.option("--test-interval", type = int, default = 5)
@click.option("--optimizer-class", type = str, default = "Adam")
@click.option("--loss-class", type = str, default = "MSE")
@click.option("--network-depth", type = int, default = 2)
@click.option("--network-width", type = int, default = 32)
def main(env_name, log_dir, gpu, **kwargs):
    #todo: add load and update parameters function
    json_file = os.path.join("configs","{}.json".format(env_name))
    if os.path.exists(json_file):
        print("updating hyper-parameters from \033[31m{}\033[0m".format(json_file))
        with open(json_file,'r') as f:
            json_dict = json.load(f)
        kwargs = update_parameters(kwargs, json_dict, True)
    #initialize device
    set_device(gpu)
    #initialize logger
    logger = LOGGER(log_dir, prefix = env_name)
    logger.log_str("logging to {}".format(log_dir))
    #initialize environment
    env = gym.make(env_name)
    #initialize model
    model  = DQNAgent(
        logger,
        env,
        **kwargs
    )
    model.train()




if __name__ == "__main__":
    main()