import os
import click
import shutil
min_log_size = 100000



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
        

def search_algo_log_dirs(curr_dir):
    dirs = os.listdir(curr_dir)
    abs_dirs = [os.path.join(curr_dir, d) for d in dirs if d != '.git']
    abs_dirs = [d for d in abs_dirs if os.path.isdir(d)]
    all_exp_log_dirs = []
    for d in abs_dirs:
        if d[-4:] == "logs":
            all_exp_log_dirs += locate_exp_log_dirs(d)
        else:
            all_exp_log_dirs += search_algo_log_dirs(d)
    return all_exp_log_dirs

def copy_exp_dirs(exp_dirs, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    for exp_dir in exp_dirs:
        source_exp_elements = exp_dir.split(os.sep)
        algo_name, task_name, exp_info = source_exp_elements[-3:]
        target_task_dir = os.path.join(target_dir, algo_name, task_name)
        if not os.path.exists(target_task_dir):
             os.makedirs(target_task_dir)
        target_exp_dir = os.path.join(target_task_dir, exp_info)
        if os.path.exists(target_exp_dir):
             shutil.rmtree(target_exp_dir, True)
        shutil.copytree(exp_dir, target_exp_dir)
        print("copied {} to {}".format(exp_dir, target_exp_dir))

def remove_dirs(exp_dirs):
    for exp_dir in exp_dirs:
        shutil.rmtree(exp_dir)
        

@click.command()
@click.option("--source_dir", type=str, default="../unstable_baselines", help="base dir to locate experiment logs")
@click.option("--target_dir", type=str, default="../logs", help="target dir to store logs")
@click.option("--delete-source",type=bool , default=False, help="whether to delete source logs")
def main(source_dir, target_dir, delete_source):
    exp_dirs = search_algo_log_dirs(source_dir)

    copy_exp_dirs(exp_dirs, target_dir)
    if delete_source:
         remove_dirs(exp_dirs)

if __name__ == "__main__":
    main()