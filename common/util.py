import torch
import json
from datetime import datetime
import os
import numpy as np
import ast
import random

device = None



def set_global_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
def set_device(gpu_id):
    global device
    if gpu_id < 0 or torch.cuda.is_available() == False:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:{}".format(gpu_id))
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    print("setting device:", device)
        

def load_config(config_path,update_args):
    with open(config_path,'r') as f:
        args_dict = json.load(f)
    #update args is tpule type, convert to dict type
    update_args_dict = {}
    for update_arg in update_args:
        key, val = update_arg.split("=")
        update_args_dict[key] = val
    args_dict = update_parameters(args_dict, update_args_dict)
    args_dict = merge_dict(args_dict, "common")
    return args_dict

def update_parameters(source_args, update_args):
    print("updating args", update_args)
    for key_path in update_args:
        target_value = update_args[key_path]
        print("key:{}\tvalue:{}".format(key_path, target_value))
        source_args = overwrite_argument(source_args, key_path, target_value)
    return source_args

def overwrite_argument(source_dict, key_path, target_value):
    key_path = key_path.split("/")
    curr_dict = source_dict
    for key in key_path[:-1]:
        if not key in curr_dict:
            #illegal path
            return source_dict
        curr_dict = curr_dict[key]
    final_key = key_path[-1]
    curr_dict[final_key] = ast.literal_eval(target_value)
    return source_dict

def soft_update_network(source_network, target_network, tau):
    for target_param, local_param in zip(target_network.parameters(),
                                        source_network.parameters()):
        target_param.data.copy_(tau*local_param.data + (1-tau)*target_param.data)

def hard_update_network(source_network, target_network):
    for target_param, local_param in zip(target_network.parameters(), 
                                         source_network.parameters()):
        target_param.data.copy_(local_param.data)


def second_to_time_str(remaining:int):
    dividers = [86400, 3600, 60, 1]
    names = ['day', 'hour', 'minute', 'second']
    results = []
    for d in dividers:
        re = int(np.floor(remaining / d))
        results.append(re)
        remaining -= re * d
    time_str = ""
    for re, name in zip(results, names):
        if re > 0 :
            time_str += "{} {}  ".format(re, name)
    return time_str

def merge_dict(source_dict, common_dict_name):
    if not  common_dict_name in source_dict:
        return source_dict
    additional_dict = source_dict[common_dict_name]
    for key in source_dict:
        if key == common_dict_name:
            continue
        if type(source_dict[key]) != dict:
            #parameter case
            continue
        for k in additional_dict:
            if k in source_dict:
                print("\033[32m Duplicate key \"{}\" when merging dict".format(k))
            source_dict[key][k] = additional_dict[k]
    return source_dict



    

if __name__ == "__main__":
    #code for testing overwriting arguments
    source_dict = {
        "a":{
            "b":1,
            "c":2
        },
        "c":0
    } 
    overwrite_argument(source_dict, "a/b", "3")
    print(source_dict)


    

