import os 
from tensorboard.backend.event_processing import event_accumulator
from tqdm import tqdm
import pandas as pd
import numpy as np


def load_tb_file(f_path, plot_interval, align_steps = True):
    re_dict = {}
    ea=event_accumulator.EventAccumulator(f_path) 
    ea.Reload()
    if align_steps:
    #return dict with only a single step list by setting non-exsisting values to np.nan
        tot_steps = []
        for key in ea.scalars.Keys():
            values = ea.scalars.Items(key)
            step_list = [v.step for v in values]
            tot_steps += step_list
        tot_steps  = list(set(tot_steps))
        tot_steps.sort()
        re_dict['steps'] = tot_steps
        #fill tot_steps
        for key in ea.scalars.Keys():
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

def find_and_load_tb_file(base_dir, plot_interval):
    file_paths = os.listdir(base_dir)
    possible_files = [f_path  for f_path in file_paths if "tfevents" in f_path]
    assert len(possible_files) <= 1
    if len(possible_files) == 0:
        return None
    else:
        return load_tb_file(os.path.join(base_dir, possible_files[0]), plot_interval)

def load_tb_logs(path, plot_interval):
    current_folder = os.getcwd()
    exp_dirs = os.listdir(path)
    print("loading from experiments")
    re_dict = {}
    for exp_dir in tqdm(exp_dirs):
        content = find_and_load_tb_file(os.path.join(current_folder, path, exp_dir), plot_interval)
        if content is not None:
            re_dict[exp_dir] = dict(
                info = get_exp_info(exp_dir),
                content = content
            )
    return re_dict

def get_exp_info(exp_name):
    infos = exp_name.split("-")
    task_name, version, info, time, pid = infos
    return dict(
        task_name = task_name + "-" + version, 
        info = info,
        time = time,
        pid = pid
    )
     

def create_log_pdframe(logs):
    task_names = []
    infos = []
    steps = []
    value_lists = {}
    print("merging and creating dataframe")
    for log_name in tqdm(logs.keys()):
        #print(logs[log_name].keys())
        info = logs[log_name]['info']  
        content = logs[log_name]['content']
        step_list = content['steps']
        steps += step_list
        task_names += [info['task_name'] for i in step_list]
        infos += [info['info'] for i in step_list]
        
        value_keys = list(content.keys())
        value_keys.remove("steps")
        for value_key in value_keys:
            value_list = content[value_key]
            if value_key not in value_lists:
                value_lists[value_key] = []
            value_lists[value_key] += value_list
    tot_dict = {}
    tot_dict['step'] = steps
    tot_dict['task_name'] = task_names
    tot_dict['info'] = infos
    for value_key in value_lists:
        tot_dict[value_key] = value_lists[value_key]
        #print(value_key, type(tot_dict[value_key][0]),tot_dict[value_key][0])
    value_keys = list(value_lists.keys())
    return pd.DataFrame(tot_dict, index = steps), value_keys

if __name__ == "__main__":
    file_path = "logs"
    re_dict = load_tb_logs(file_path)
    print("loaded", re_dict.keys())