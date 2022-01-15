import os
import click
from unstable_baselines.common.logger import Logger
from unstable_baselines.baselines.ppo.trainer import PPOTrainer
from unstable_baselines.baselines.ppo.agent import PPOAgent
from unstable_baselines.common.util import set_device_and_logger, load_config, set_global_seed
from unstable_baselines.common.env_wrapper import get_env, BaseEnvWrapper
from unstable_baselines.common.rollout import RolloutBuffer

@click.command(context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
))
@click.argument("config-path",type=str, required=True)
@click.option("--log-dir", default=os.path.join("logs", "ppo"))
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
    logger = Logger(log_dir, env_name, prefix = info, print_to_terminal=print_log)

    #set device and logger
    set_device_and_logger(gpu, logger)

    #save args
    logger.log_str_object("parameters", log_dict = args)

    #initialize environment
    logger.log_str("Initializing Environment")
    train_env = get_env(env_name)
    eval_env = get_env(env_name)
    state_space = train_env.observation_space
    action_space = train_env.action_space

    #initialize agent
    logger.log_str("Initializing Agent")
    agent = PPOAgent(state_space, action_space, **args['agent'])

    #initailize rollout buffer
    rollout_buffer = RolloutBuffer(state_space, action_space, **args['buffer'])
    
    #initialize trainer
    logger.log_str("Initializing Trainer")
    trainer  = PPOTrainer(
        agent,
        train_env,
        eval_env,
        rollout_buffer,
        **args['trainer']
    )

    
    logger.log_str("Started training")
    trainer.train()


if __name__ == "__main__":
    main()