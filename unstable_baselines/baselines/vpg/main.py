import os
import sys
from typing import Any

import gym
import click
from torch._C import default_generator

from unstable_baselines.common.logger import Logger
from unstable_baselines.baselines.vpg.agent import VPGAgent
from unstable_baselines.baselines.vpg.trainer import VPGTrainer
from unstable_baselines.common.util import set_device_and_logger
from unstable_baselines.common.util import load_config
from unstable_baselines.common.util import set_global_seed
from unstable_baselines.common.buffer import OnlineBuffer
from unstable_baselines.common.env_wrapper import BaseEnvWrapper, get_env

# debug
from unstable_baselines.common import util


@click.command(context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
))
@click.argument("config-path",type=str, required=True)
@click.option("--log-dir", default=os.path.join("logs", "vpg"))
@click.option("--gpu", type=int, default=0)
@click.option("--print-log", type=bool, default=True)
@click.option("--seed", type=int, default=35)
@click.option("--info", type=str, default="")
@click.argument("args", nargs=-1)
def main(config_path: str,
         log_dir: str,
         gpu: int,
         print_log: bool,
         seed: int,
         info: str,
         args: Any
    ):

    # load and update parameters function
    args = load_config(config_path, args)

    # set global seed
    set_global_seed(seed)

    # initialize logger
    env_name = args['env_name']
    logger = Logger(log_dir, env_name, prefix = info, print_to_terminal=print_log)

    # set device and logger
    set_device_and_logger(gpu, logger)

    # save args
    logger.log_str_object("parameter", log_dict=args)

    # initialize environment
    logger.log_str("Initializing Environment")
    train_env = get_env(env_name)
    eval_env = get_env(env_name)
    state_space = train_env.observation_space
    action_space = train_env.action_space

    # initialize buffer
    buffer = OnlineBuffer(state_space, action_space, **args['buffer'])

    # initialize agent
    logger.log_str("Initializing Agent")
    agent = VPGAgent(state_space, action_space, **args['agent'])

    # initialize trainer
    logger.log_str("Initializing Trainer")
    trainer = VPGTrainer(
        agent,
        train_env,
        eval_env,
        buffer,
        **args['trainer']
    )

   # start training
    logger.log_str("Started training") 
    trainer.train()


if __name__ == '__main__':
    main()
