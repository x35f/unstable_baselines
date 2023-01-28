import os
import click
import subprocess
from unstable_baselines.common.util import load_config, set_device_and_logger
from unstable_baselines.common.logger import Logger
from device_wrapper import DeviceInstance

from time import sleep


def generate_commands(algos, tasks, seeds, log_dir, overwrite_args):
    command_strs = []
    for algo, algo_dir in algos.items():
        algo_main_file_path = os.path.join(algo_dir, "main.py")
        algo_overwrite_args = overwrite_args.get(algo, {})
        for task in tasks:
            task_config_path = os.path.join(algo_dir, 'configs', task+".py")
            task_overwrite_args = algo_overwrite_args.get(task, {})
            for seed in seeds:
                overwrite_arg_str = ""
                for key, value in task_overwrite_args.items():
                    overwrite_arg_str += "{}={} ".format(key, value)
                for seed  in seeds:
                    command_str = "python {} {} --seed {} --log_dir {} {}".format(algo_main_file_path,task_config_path, seed, log_dir, overwrite_arg_str)
                    command_strs.append(command_str)
    return command_strs

@click.command()
@click.argument("config-path", type=str, required=True)
@click.option("--log-path", type=str, default="logs")
def main(config_path, log_path):

    config = load_config(config_path)
    logger = Logger(log_path, config_path.split(os.sep)[-1], 0)
    set_device_and_logger(-1, logger)
    logger.log_str_object("config", log_dict=config)
    algos = config['algos']
    tasks = config['tasks']
    seeds = config['seeds']
    overwrite_args = config['overwrite_args']
    exp_log_dir = config['log_dir']
    gpu_ids = config['gpu_ids']

    commands = generate_commands(algos, tasks, seeds, exp_log_dir, overwrite_args)
    for command in commands:
        print(command)
    exit(0)
    # initialized device controller
    device = DeviceInstance(gpu_ids, config['estimated_system_memory_per_exp'], config['estimated_gpu_memory_per_exp'], config['max_exps_per_gpu'])

    #run commands one by one
    for i, command in enumerate(commands):
        successed = False
        logger.log_str("Preparing to run command {}/{}: {}".format(i+1, len(commands), command))
        while not successed:
            ret = device.run(command)
            
            if not isinstance(ret, int):
                logger.log_str("Command {} start failed, error message: {}".format(i+1, ret))
                exit(0)
            if ret > 0:
                successed = True
                logger.log_str("Started command {} with pid {}".format(i+1, ret))
                break
            elif ret == -1:
                logger.log_str("Command {} failed because system memory is not enough".format(i+1, ret))
            elif ret == -2:
                logger.log_str("Command {} failed because max instances num has exceeded".format(i+1, ret))
            elif ret == -3:
                logger.log_str("Command {} failed because of insufficient gpu memory".format(i+1, ret))
            else:
                exit(0)
            sleep(10)


      

if __name__ == "__main__":
    main()