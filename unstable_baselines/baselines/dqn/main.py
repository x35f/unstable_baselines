import os
import sys
# sys.path.append(os.path.join(os.getcwd(), './'))
# sys.path.append(os.path.join(os.getcwd(), '../..'))
import gym
import click
from unstable_baselines.common.logger import Logger
from unstable_baselines.baselines.dqn.agent import DQNAgent
from unstable_baselines.baselines.dqn.trainer import DQNTrainer
from unstable_baselines.common.util import set_device_and_logger, load_config, set_global_seed
from unstable_baselines.common.buffer import ReplayBuffer
from unstable_baselines.common.env_wrapper import ATARI_ENVS, get_env, AtariWrapper, wrap_atari_env
from tqdm import tqdm
from functools import partialmethod

@click.command(context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
))
@click.argument("config-path",type=str, required=True)
@click.option("--log-dir", default=os.path.join("logs", "dqn"))
@click.option("--gpu", type=int, default=-1)
@click.option("--print-log", type=bool, default=True)
@click.option("--enable-pbar", type=bool, default=True)
@click.option("--seed", type=int, default=35)
@click.option("--info", type=str, default="")
@click.option("--load_path", type=str, default="")
@click.argument('args', nargs=-1)
def main(config_path, log_dir, gpu, print_log, enable_pbar, seed, info, load_path, args):
    #todo: add load and update parameters function
    args = load_config(config_path, args)

    #silence tqdm progress bar output
    if not enable_pbar:
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

    #set global seed
    set_global_seed(seed)

    #initialize logger
    env_name = args['env_name']
    logger = Logger(log_dir, env_name, seed, info_str = info, print_to_terminal=print_log)

    #set device and logger
    set_device_and_logger(gpu, logger)

    #save args
    logger.log_str_object("parameters", log_dict = args)

    #initialize environment
    logger.log_str("Initializing Environment")
    train_env = get_env(env_name, seed=seed)
    eval_env = get_env(env_name, seed=seed)
    if env_name in ATARI_ENVS:
        train_env = wrap_atari_env(train_env, **args['env'])
        eval_env = wrap_atari_env(eval_env,  **args['env'])
    observation_space = train_env.observation_space
    action_space = train_env.action_space

    #initialize buffer
    logger.log_str("Initializing Buffer") 
    buffer = ReplayBuffer(observation_space, action_space, **args['buffer'])

    #initialize model
    logger.log_str("Initializing Agent")
    agent  = DQNAgent(
        observation_space, action_space,
        **args['agent']
    )

    #initialize trainer
    logger.log_str("Initializing Trainer")
    trainer  = DQNTrainer(
        agent,
        train_env,
        eval_env,
        buffer,
        load_path,
        **args['trainer']
    )

    
    logger.log_str("Started training")
    trainer.train()




if __name__ == "__main__":
    main()