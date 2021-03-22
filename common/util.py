import torch
import json
from datetime import datetime
import os
import numpy as np
import random
import gym

device = None

def set_device(gpu_id):
    global device
    if gpu_id < 0 or torch.cuda.is_available() == False:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:{}".format(gpu_id))
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    print("setting device:", device)
        

def load_config(config_path, update_args):
    with open(config_path,'r') as f:
        args_dict = json.load(f)
    args_dict = update_parameters(args_dict, update_args)
    return args_dict

def update_parameters(source_args, new_args):
    for key in new_args:
        path = key.split("/")
        #todo override source arg through path
    return source_args


def soft_update_network(source_network, target_network, tau):
    for target_param, local_param in zip(target_network.parameters(),
                                        source_network.parameters()):
        target_param.data.copy_(tau*local_param.data + (1-tau)*target_param.data)

def hard_update_network(source_network, target_network):
    for target_param, local_param in zip(target_network.parameters(), 
                                         source_network.parameters()):
        target_param.data.copy_(local_param.data)


def second_to_time_str(remaining:int):
    year, month, day, hour, minute, second = 0, 0, 0, 0, 0, 0
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



    

