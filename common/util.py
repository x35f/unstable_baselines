import torch

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
        device = torch.device("cuda:0")
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        

def load_config(config_path, **kwargs):
    with open(config_path,'r') as f:
        args_dict = json.load(f)
    args_dict = update_parameters(args_dict, kwargs)
    return args_dict


def soft_update_network(source_network, target_network, tau):
    for target_param, local_param in zip(target_network.parameters(),
                                        source_network.parameters()):
        target_param.data.copy_(tau*local_param.data + (1-tau)*target_param.data)



    

