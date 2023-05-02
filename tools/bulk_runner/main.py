import os
import click
import subprocess
from unstable_baselines.common.util import load_config, set_device_and_logger
from unstable_baselines.common.logger import Logger
from device_wrapper import DeviceInstance

from time import sleep

import unstable_baselines as usb

#disable tqdm output
from tqdm import tqdm
from functools import partialmethod
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


import signal
import time


device:DeviceInstance = None

def handler(signum, frame):
    res = input("Ctrl-c was pressed. Do you really want to kill all exps? y/n ")
    if res == 'y':
        device.kill_all_exps()
        exit(1)

signal.signal(signal.SIGINT, handler)


usb_root_path = usb.__path__[0]

def generate_commands(algos, tasks, seeds, log_dir, overwrite_args, redirected_exp_output_path):
    command_strs = []
    redirected_exp_output_dir = os.path.join(redirected_exp_output_path, "exp_output")
    if not os.path.exists(redirected_exp_output_dir):
        os.makedirs(redirected_exp_output_dir)
    
    for algo, algo_dir in algos.items():
        algo_main_file_path = os.path.join(usb_root_path, algo_dir, "main.py")
        algo_overwrite_args = overwrite_args.get(algo, {})
        algo_log_dir = os.path.join(log_dir, algo)
        for task in tasks:
            task_config_path = os.path.join("unstable_baselines", algo_dir, 'configs', task+".py")
            task_overwrite_args = algo_overwrite_args.get(task, {})
            for seed in seeds:
                overwrite_arg_str = ""
                for key, value in task_overwrite_args.items():
                    overwrite_arg_str += "{}={} ".format(key, value)
                redirected_exp_output_file_path = os.path.join(redirected_exp_output_dir, "{}_{}_{}.txt".format(algo, task.replace("/","_"), seed))
                command_str = "python {} {} --enable-pbar False --seed {} --log-dir {} {}".format(algo_main_file_path,task_config_path, seed, algo_log_dir, overwrite_arg_str)
                command_strs.append([command_str, redirected_exp_output_file_path])
    return command_strs

@click.command()
@click.argument("config-path", type=str, required=True)
@click.option("--log-path", type=str, default="logs")
def main(config_path, log_path):

    config = load_config(config_path)
    logger = Logger(log_path, config_path.split(os.sep)[-1].replace(".py", ""), 0)
    set_device_and_logger(-1, logger)
    logger.log_str_object("config", log_dict=config)
    algos = config['algos']
    tasks = config['tasks']
    seeds = config['seeds']
    overwrite_args = config['overwrite_args']
    exp_log_dir = config['log-dir']
    gpu_ids = config['gpu_ids']
    refresh_interval = config['refresh_interval']

    commands = generate_commands(algos, tasks, seeds, exp_log_dir, overwrite_args, logger.log_path)

    # initialized device controller
    global device
    device = DeviceInstance(gpu_ids, config['estimated_system_memory_per_exp'], config['estimated_gpu_memory_per_exp'], config['max_exps_per_gpu'])
    #try to run commands one by one
    for i, (command, exp_output_path) in enumerate(commands):
        successed = False
        logger.log_str("Preparing to run command {}/{}: {}".format(i+1, len(commands), command))
        while not successed:
            ret = device.run(command, exp_output_path)
            
            if not isinstance(ret, int):
                logger.log_str("Command {} start failed, error message: {}".format(i+1, ret))
                exit(0)
            if ret > 0:
                successed = True
                logger.log_str("Started command {} with pid {}".format(i+1, ret))
                break
            elif ret == -1:
                logger.log_str("Command {} hanged because of insufficient system memory".format(i+1, ret))
            elif ret == -2:
                logger.log_str("Command {} hanged because max instances num has exceeded".format(i+1, ret))
            elif ret == -3:
                logger.log_str("Command {} hanged because of insufficient gpu memory".format(i+1, ret))
            else:
                exit(0)
            sleep(refresh_interval)

    #do not close this program until all experiments are finished
    while True:
        running_exp_num =  device.running_instance_num()
        if running_exp_num > 0:
            logger.log_str("{} exps are still running".format(running_exp_num))
        else:
            logger.log_str("all {} experiments have finished".format(len(commands)))
            break
        sleep(refresh_interval)



      

if __name__ == "__main__":
    main()