import os 
from tensorboard.backend.event_processing import event_accumulator
from tqdm import tqdm
import pandas as pd
import numpy as np

def load_logs(log_dir, config):
    algos = config['algos']
    tasks = config['tasks']
    keys = config['key_mapping'].keys()
    plot_interval = config['plot_interval']
    smooth_length = config['smooth_length']
    tb_logs = {}
    for algo in tqdm(algos):
        tb_logs[algo] = {}
        for task in tasks:
            tb_logs[algo][task] = {}
            task_dir = os.path.join(log_dir, algo, task)
            if not os.path.exists(task_dir):
                continue
            exp_dirs = os.listdir(task_dir)
            for exp_dir in exp_dirs:
                exp_relative_path = os.path.join(task_dir, exp_dir)
                tb_logs[algo][task][exp_dir] = load_tb_logs(exp_relative_path, keys, plot_interval, max_timestep=tasks[task], smooth_length=smooth_length)
    return tb_logs

def load_tb_logs(exp_path, keys, plot_interval, align_steps = True, max_timestep=999999999999, smooth_length=1):
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
            step_list = [v.step for v in values if v.step <= max_timestep]
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
            clipped_step_list = [s for s in step_list if s <= max_timestep]
            clipped_value_list = [v for v,s in zip(value_list, step_list) if s <= max_timestep]
            step_list = clipped_step_list
            value_list = clipped_value_list
            #apply smoothing
            convkernel = np.ones(2 * smooth_length+1)
            value_list = np.convolve(value_list, convkernel,mode='same') / np.convolve(np.ones_like(value_list), convkernel, mode='same')
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
                    assert(len(value_list) ==len(step_list))
                    value_lists[KEY_MAPPING[key]] += value_list
    tot_dict = {}
    tot_dict['timestep'] = steps
    tot_dict['task_name'] = task_names 
    tot_dict['algo_name'] = algo_names
    tot_dict['exp_name'] = exp_names
    for value_key in value_lists:
        tot_dict[value_key] = value_lists[value_key]
    
    value_keys = list(value_lists.keys())
    return pd.DataFrame(tot_dict), value_keys