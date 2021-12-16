import os
import sys
# sys.path.append(os.path.join(os.getcwd(), '..'))
import gym
import click
from unstable_baselines.common.logger import Logger
from unstable_baselines.model_based_rl.mbpo.trainer import MBPOTrainer
from unstable_baselines.model_based_rl.mbpo.agent import MBPOAgent
from unstable_baselines.common.util import set_device_and_logger, load_config, set_global_seed
from unstable_baselines.common.buffer import PrioritizedReplayBuffer, ReplayBuffer
from unstable_baselines.common.env_wrapper import get_env, BaseEnvWrapper
from unstable_baselines.common import util

@click.command(context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
))
@click.argument("config-path",type=str, required=True)
@click.option("--log-dir", default="logs/mbpo")
@click.option("--gpu", type=int, default=-1)
@click.option("--print-log", type=bool, default=True)
@click.option("--seed", type=int, default=30)
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
    env = get_env(env_name)
    #env = BaseEnvWrapper(env, **args['env'])
    eval_env = get_env(env_name)
    #eval_env = BaseEnvWrapper(eval_env, **args['env'])
    state_space = env.observation_space
    action_space = env.action_space

    #initialize buffer
    logger.log_str("Initializing Buffer")
    env_buffer = ReplayBuffer(state_space, action_space, **args['buffer'])
    model_buffer = ReplayBuffer(state_space, action_space, **args['buffer'])

    #initialize agent
    logger.log_str("Initializing Agent")
    agent = MBPOAgent(state_space, action_space, env_name=env_name, **args['agent'])

    #initialized generator 
    #initialize trainer
    logger.log_str("Initializing Trainer")
    trainer  = MBPOTrainer(
        agent,
        env,
        eval_env,
        env_buffer,
        model_buffer,
        **args['trainer']
    )
    
    logger.log_str("Started training")
    trainer.train()


if __name__ == "__main__":
    main()