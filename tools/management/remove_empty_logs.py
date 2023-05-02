import os
from typing_extensions import Required
import click
import shutil
min_log_size = 100000

def get_dir_size(dir):
    size = 0
    for root, dirs, files in os.walk(dir):
        size += sum([os.path.getsize(os.path.join(root, name)) for name in files])
    return size

def locate_exp_log_dirs(log_dir):
    algo_dirs = os.listdir(log_dir)
    for algo_dir in algo_dirs:
        algo_abs_dir = os.path.join(log_dir, algo_dir)
        env_dirs = os.listdir(algo_abs_dir)
        env_abs_dirs = [os.path.join(algo_abs_dir, d) for d in env_dirs]
        env_abs_dirs = [d for d in env_abs_dirs if os.path.isdir(d)]
        exp_abs_dirs = []
        for env_abs_dir in env_abs_dirs:
            exp_dirs = os.listdir(env_abs_dir)
            for exp_dir in exp_dirs:
                exp_abs_dir = os.path.join(env_abs_dir, exp_dir)
                if os.path.isdir(exp_abs_dir):
                    exp_abs_dirs.append(exp_abs_dir)
    return exp_abs_dirs

def remove_empty_logs(log_dirs):
    for log_dir in log_dirs:
        log_dir_size = get_dir_size(log_dir)
        if log_dir_size < min_log_size:
            shutil.rmtree(log_dir)
            print("removed dir {}, size: {}".format(log_dir, log_dir_size))
        

def search_algo_log_dirs(curr_dir):
    dirs = os.listdir(curr_dir)
    #print(curr_dir, dirs)
    abs_dirs = [os.path.join(curr_dir, d) for d in dirs if d != '.git']
    abs_dirs = [d for d in abs_dirs if os.path.isdir(d)]
    for d in abs_dirs:
        if d[-4:] == "logs":
            exp_log_dirs = locate_exp_log_dirs(d)
            remove_empty_logs(exp_log_dirs)
        else:
            search_algo_log_dirs(d)

@click.command()
@click.argument("base_dir", type=str)
def main(base_dir):
    search_algo_log_dirs(base_dir)

if __name__ == "__main__":
    main()