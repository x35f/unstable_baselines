import os
import click
import shutil

min_log_size = 100000

def locate_exp_log_dirs(log_dir):
    algo_dirs = os.listdir(log_dir)
    all_exp_dirs = []
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
        all_exp_dirs += exp_abs_dirs
    return all_exp_dirs

def rename_logs(log_dirs):
    for log_dir in log_dirs:
        log_elements = log_dir.split("/")
        exp_info = log_elements[-1]
        year, month = exp_info.split('-')[:2]
        try:
            log_elements = log_dir.split("/")
            exp_info = log_elements[-1].replace("(", "-").replace(")","-").replace(":","-")
            if "--" in exp_info:
                exp_info = "2022-" + exp_info[exp_info.index("--")+2:]

            has_suffix = "_" in exp_info
            if has_suffix:
                suffix_start = exp_info.index("_") + 1
                suffix_str = exp_info[suffix_start:]
                exp_info = exp_info[:suffix_start - 1]
            else:
                suffix_str = ""

            dash_counts = exp_info.count("-")
            if dash_counts == 4:
                exp_elements = exp_info.split("-")
                if int(exp_elements[0]) < 6:
                    exp_elements.insert(0,'2022')
                else:
                    exp_elements.insert(0,'2021')
                exp_elements.insert(5,'00')
                exp_info = "-".join(exp_elements)
            if dash_counts == 5:
                exp_elements = exp_info.split("-")
                exp_elements.insert(5,'00')
                exp_info = "-".join(exp_elements)
            elif dash_counts == 6:
                pass
            if suffix_str != "":
                exp_info = exp_info + "_" + suffix_str
            log_elements[-1] = exp_info
            new_log_dir = os.sep.join(log_elements)
            if "--" in new_log_dir:
                new_log_dir = new_log_dir.replace("--",'-00-')
            # if new_log_dir != log_dir:
            #     print(log_dir, '\n', new_log_dir, '\n')
            if new_log_dir != log_dir:
                os.rename(log_dir, new_log_dir)
        except:
            print(log_dir)
            exit(0)

        

def search_algo_log_dirs(curr_dir):
    dirs = os.listdir(curr_dir)
    #print(curr_dir, dirs)
    abs_dirs = [os.path.join(curr_dir, d) for d in dirs if d != '.git']
    abs_dirs = [d for d in abs_dirs if os.path.isdir(d)]
    for d in abs_dirs:
        if d[-4:] == "logs":
            exp_log_dirs = locate_exp_log_dirs(d)
            rename_logs(exp_log_dirs)
        else:
            search_algo_log_dirs(d)

@click.command()
@click.argument("base_dir", type=str)
def main(base_dir):
    search_algo_log_dirs(base_dir)

if __name__ == "__main__":
    main()