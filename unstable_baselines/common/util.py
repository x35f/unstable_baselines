import torch
import json
from datetime import datetime
import os
import numpy as np
import ast
import random
import scipy.signal

device = None
logger = None


def set_global_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def set_device_and_logger(gpu_id, logger_ent):
    global device, logger
    if gpu_id < 0 or torch.cuda.is_available() == False:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:{}".format(gpu_id))
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    print("setting device:", device)
    logger = logger_ent

def load_config(config_path,update_args):
    default_config_path_elements = config_path.split("/")
    default_config_path_elements[-1] = "default.json"
    default_config_path = os.path.join(*default_config_path_elements)
    with open(default_config_path, 'r') as f:
        default_args_dict = json.load(f)
    with open(config_path,'r') as f:
        args_dict = json.load(f)

    #update args is tpule type, convert to dict type
    update_args_dict = {}
    for update_arg in update_args:
        key, val = update_arg.split("=")
        update_args_dict[key] = ast.literal_eval(val)
    
    #update env specific args to default 
    default_args_dict = update_parameters(default_args_dict, update_args_dict)
    args_dict = merge_dict(default_args_dict, args_dict)
    if 'common' in args_dict:
        for sub_key in args_dict:
            if type(args_dict[sub_key]) == dict:
                args_dict[sub_key] = merge_dict(args_dict[sub_key], default_args_dict['common'], "common")
    return args_dict

def merge_dict(source_dict, update_dict, ignored_dict_name=""):
    for key in update_dict:
        if key == ignored_dict_name:
            continue
        if key not in source_dict:
            #print("\033[32m new arg {}: {}\033[0m".format(key, update_dict[key]))
            source_dict[key] = update_dict[key]
        else:
            assert type(source_dict[key]) == type(update_dict[key])
            if type(update_dict[key]) == dict:
                source_dict[key] = merge_dict(source_dict[key], update_dict[key], ignored_dict_name)
            else:
                print("updated {} from {} to {}".format(key, source_dict[key], update_dict[key]))
                source_dict[key] = update_dict[key]
    return source_dict

def update_parameters(source_args, update_args):
    print("updating args", update_args)
    #command line overwriting case, decompose the path and overwrite the args
    for key_path in update_args:
        target_value = update_args[key_path]
        print("key:{}\tvalue:{}".format(key_path, target_value))
        source_args = overwrite_argument_from_path(source_args, key_path, target_value)
    return source_args


def overwrite_argument_from_path(source_dict, key_path, target_value):
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


def discount_cum_sum(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def minibatch_rollout(data, network, batch_size = 256):
    data_size = len(data)
    num_batches = np.ceil(data_size/batch_size)
    result = []

    for i in range(num_batches - 1):
        result.append(network(data[ i * batch_size: (i + 1) * batch_size]))
    result.append(network(data[(num_batches - 1) * batch_size:]))
    result = torch.cat(result)
    return result 

if __name__ == "__main__":
    #code for testing overwriting arguments
    # source_dict = {
    #     "a":{
    #         "b":1,
    #         "c":2
    #     },
    #     "c":0
    # } 
    # overwrite_argument(source_dict, "a/b", "3")
    # print(source_dict)

    #code for testing discount_cum_sum funciton
    cum_list = [1,1,1,1,1,1,1,1,1]
    discount_factor=0.9

    re = discount_cum_sum(cum_list, discount_factor)
    print(re)


    

