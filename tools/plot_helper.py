import os 
from tensorboard.backend.event_processing import event_accumulator
from tqdm import tqdm
import pandas as pd
import numpy as np

def load_logs(log_dir, algos, tasks, keys, plot_interval):
    file_paths = os.listdir(log_dir)
    tb_logs = {}
    for algo in tqdm(algos):
        tb_logs[algo] = {}
        for task in tasks:
            tb_logs[algo][task] = {}
            task_dir = os.path.join(log_dir, algo, task)
            exp_dirs = os.listdir(task_dir)
            for exp_dir in exp_dirs:
                exp_relative_path = os.path.join(task_dir, exp_dir)
                tb_logs[algo][task][exp_dir] = load_tb_logs(exp_relative_path, keys, plot_interval)
    return tb_logs

def load_tb_logs(exp_path, keys, plot_interval, align_steps = True):
    file_paths = os.listdir(exp_path) 
    event_files = [f_path  for f_path in file_paths if "tfevents" in f_path]
    assert len(event_files) == 1
    event_file = event_files[0]
    event_file_path = os.path.join(exp_path, event_file)
    re_dict = {}
    ea=event_accumulator.EventAccumulator(event_file_path) 
    ea.Reload()
    if align_steps:
    #return dict with only a single step list by setting non-exsisting values to np.nan
        tot_steps = []
        for key in ea.scalars.Keys():
            if key not in keys:
                continue
            values = ea.scalars.Items(key)
            step_list = [v.step for v in values]
            tot_steps += step_list
        tot_steps  = list(set(tot_steps))
        tot_steps.sort()
        re_dict['steps'] = tot_steps
        #fill tot_steps
        for key in ea.scalars.Keys():
            if key not in keys:
                continue
            values = ea.scalars.Items(key)
            step_list = [v.step for v in values][::plot_interval]
            #step_list.sort()
            value_list = [v.value for v in values][::plot_interval]
            expanded_value_list = [np.nan for i in range(len(tot_steps))]
            i, j = 0,0
            while(i < len(step_list) and j < len(tot_steps)):
                if step_list[i] == tot_steps[j]:
                    expanded_value_list[j] = value_list[i]
                    i += 1
                    j += 1
                elif step_list[i] > tot_steps[j]:
                    j += 1
                else:
                    assert 0
            re_dict[key] = expanded_value_list
    else:
        for key in ea.scalars.Keys():
            values = ea.scalars.Items(key)
            step_list = [v.step for v in values]
            #step_list.sort()
            value_list = [v.value for v in values]
            re_dict[key] = {
                "steps": step_list,
                "values": value_list
            }
    return re_dict


def create_log_pdframe(logs, KEY_MAPPING):
    #create a pd table of keys: alg_name, task_name, exp_info, step, key0, key1, ......
    task_names = []
    algo_names = []
    exp_names = []
    steps = []
    value_lists = {}

    print("merging and creating dataframe")
    for algo_name in tqdm(logs.keys()):
        algo_logs = logs[algo_name]
        for task_name in algo_logs.keys():
            algo_task_logs = algo_logs[task_name]
            for exp_name in algo_task_logs.keys():
                exp_logs = algo_task_logs[exp_name]
                step_list = exp_logs['steps']
                #add exp info to list
                steps += step_list
                task_names += [task_name for i in step_list]
                algo_names += [algo_name for i in step_list]
                exp_names += [exp_name for i in step_list]

                #read key and values
                value_keys = list(exp_logs.keys())
                value_keys.remove('steps')
                for key in value_keys:
                    value_list = exp_logs[key]
                    if KEY_MAPPING[key] not in value_lists:
                        value_lists[KEY_MAPPING[key]] = []
                        #print(len(steps), len(step_list))
                        assert len(steps) == len(step_list)
                    value_lists[KEY_MAPPING[key]] += value_list
    tot_dict = {}
    print(len(steps), len(task_names), len(algo_names), len(exp_names))
    tot_dict['step'] = steps
    tot_dict['task_name'] = task_names 
    tot_dict['algo_name'] = algo_names
    tot_dict['exp_name'] = exp_names
    for value_key in value_lists:
        print(value_key, len( value_lists[value_key]))
        tot_dict[value_key] = value_lists[value_key]
    
    value_keys = list(value_lists.keys())
    return pd.DataFrame(tot_dict), value_keys

if __name__ == "__main__":
    file_path = "logs"
    re_dict = load_tb_logs(file_path)
    print("loaded", re_dict.keys())