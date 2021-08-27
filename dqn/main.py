import os
import sys
sys.path.append(os.path.join(os.getcwd(), '..'))
import gym
import click
from common.logger import Logger
from dqn.agent import DQNAgent
from dqn.trainer import DQNTrainer
from common.util import set_device_and_logger, update_parameters, load_config, set_global_seed
from  common import util
from common.buffer import TDReplayBuffer


@click.command(context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
))
@click.argument("config-path",type=str)
@click.option("--log-dir", default="logs")
@click.option("--gpu", type=int, default=-1)
@click.option("--print-log", type=bool, default=True)
@click.option("--seed", type=int, default=35)
@click.option("--info", type=str, default="")
@click.argument('args', nargs=-1)
def main(config_path, log_dir, gpu, print_log, seed, info, args):
    #todo: add load and update parameters function
    args = load_config(config_path, args)

    #set global seed
    set_global_seed(seed)

    #initialize logger
    env_name = args['env_name']
    logger = Logger(log_dir, prefix = env_name+"-"+info, print_to_terminal=print_log)
    logger.log_str("logging to {}".format(logger.log_path))

    #set device and logger
    set_device_and_logger(gpu, logger)

    #save args
    logger.log_str_object("parameters", log_dict = args)

    #initialize environment
    logger.log_str("Initializing Environment")
    env = gym.make(env_name)
    eval_env = gym.make(env_name)
    state_space = env.observation_space
    action_space = env.action_space

    #initialize buffer
    logger.log_str("Initializing Buffer") 
    buffer = TDReplayBuffer(state_space, action_space, **args['buffer'])

    #initialize model
    logger.log_str("Initializing Agent")
    agent  = DQNAgent(
        state_space, action_space,
        **args['agent']
    )

    #initialize trainer
    logger.log_str("Initializing Trainer")
    trainer  = DQNTrainer(
        agent,
        env,
        eval_env,
        buffer,
        logger,
        **args['trainer']
    )

    
    logger.log_str("Started training")
    trainer.train()




if __name__ == "__main__":
    main()