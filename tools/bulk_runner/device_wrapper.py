import pynvml
from unstable_baselines.common import util
import psutil
import subprocess
import random
pynvml.nvmlInit()


class DeviceInstance:
    def __init__(self, gpu_ids, estimated_system_memory_per_exp, estimated_gpu_memory_per_exp, max_exps_per_gpu):
        #initialize gpu instances
        self.gpu_instances = []
        self.running_exps = {}
        for gpu_id in gpu_ids:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            gpu_instance = GpuInstance(gpu_id, handle)
            self.gpu_instances.append(gpu_instance)
            self.running_exps[gpu_id] = []
            
        self.estimated_system_memory_per_exp = estimated_system_memory_per_exp
        self.estimated_gpu_memory_per_exp = estimated_gpu_memory_per_exp
        self.max_exps_per_gpu = max_exps_per_gpu
    
    def run(self, command):
        self.refresh()

        #check system memory availability
        system_mem_info = psutil.virtual_memory()
        available_memory = system_mem_info.available / 1024 / 1024
        if available_memory < self.estimated_system_memory_per_exp:
            return -1

        #check gpu running instances availability
        gpu_available = False
        for gpu_id, running_exps in self.running_exps.items():
            if len(running_exps) < self.max_exps_per_gpu:
                gpu_available = True
                break
        if not gpu_available:
            return -2

        #check gpu memory availability
        available_gpus = []
        for gpu_instance in self.gpu_instances:
            if gpu_instance.get_available_memory() > self.estimated_gpu_memory_per_exp:
                available_gpus.append(gpu_instance)
        if len(available_gpus) == 0:
            return -3

        # add gpu id to parameters and run command
        gpu_instance = random.choice(available_gpus) # randomly choose one availabel gpu
        command_elements = command.split(" ")
        command_elements = command_elements[:3] + ['--gpu', str(gpu_instance.id)] + command_elements[3:]
        final_command = " ".join(command_elements)
        try:
            pid = subprocess.Popen(final_command)
        except Exception as e:
            return e
        self.running_exps.append([pid])
        return pid

    def refresh(self):
        for gpu_id, gpu_running_exps in self.running_exps.items():
            updated_exps = []
            for instance in enumerate(gpu_running_exps):
                ret = instance.poll()
                if ret is None:
                    continue
                else:
                    updated_exps.append(instance)
            self.running_exps[gpu_id] = updated_exps
        

class GpuInstance:
    def __init__(self, id, handle):
        self.id = id
        self.running_instances = []
        self.handle = handle

    
    def get_available_memory(self):
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        free_mem = meminfo.free / 1024 / 1024
        return free_mem
    