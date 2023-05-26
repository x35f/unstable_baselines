import os
from datetime import datetime
import pickle
import json
from torch.utils.tensorboard import SummaryWriter
from abc import ABC, abstractmethod

class BaseLogger(object):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def log_str(self):
        pass

    @abstractmethod
    def log_var(self):
        pass

    
class Logger(BaseLogger):
    def __init__(self, log_path, env_name, seed, info_str="",  warning_level = 3, print_to_terminal = True):
        unique_path = self.make_simple_log_path(info_str, seed)
        log_path = os.path.join(log_path, env_name, unique_path)
        self.log_path = log_path
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        self.tb_writer = SummaryWriter(log_path)
        self.log_file_path = os.path.join(log_path,"logs.txt")
        self.print_to_terminal = print_to_terminal
        self.warning_level = warning_level
        self.log_str("logging to {}".format(self.log_path))
        
    def make_simple_log_path(self, info_str, seed):
        now = datetime.now()
        time_str = now.strftime("%Y-%m-%d-%H-%M")
        pid_str = os.getpid()
        if info_str != "":
            return "{}-{}-{}_{}".format(time_str, seed, pid_str, info_str)
        else:
            return "{}-{}-{}".format(time_str, seed, pid_str)

    @property
    def log_dir(self):
        return self.log_path
        
    def log_str(self, content, level = 4):
        if level < self.warning_level:
            return
        now = datetime.now()
        time_str = now.strftime("%Y-%m-%d %H:%M:%S")
        if self.print_to_terminal:
            print("\033[32m{}\033[0m:\t{}".format(time_str, content))
        with open(self.log_file_path,'a+') as f:
            f.write("{}:\t{}\n".format(time_str, content))

    def log_var(self, name, val, timestamp):
        self.tb_writer.add_scalar(name, val, timestamp)

    def log_str_object(self, name: str, log_dict: dict = None, log_str: str = None):
        if log_dict!=None:
            log_str = json.dumps(log_dict, indent=4)            
        elif log_str!= None:     
            pass
        else:
            assert 0
        if name[-4:] != ".txt":
            name += ".txt"
        target_path = os.path.join(self.log_path, name)
        with open(target_path,'w+') as f:
            f.write(log_str)
        self.log_str("saved {} to {}".format(name, target_path))
