import pynvml
from unstable_baselines.common import util
import psutil
import subprocess
import random
import os
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
    
    def run(self, command, output_file_path):
        self.refresh()

        #check system memory availability
        system_mem_info = psutil.virtual_memory()
        available_memory = system_mem_info.available / 1024 / 1024
        if available_memory < self.estimated_system_memory_per_exp:
            return -1

        #check gpu running instances availability
        available_gpus = []
        for gpu_instance in self.gpu_instances:
            if len(self.running_exps[gpu_instance.id]) < self.max_exps_per_gpu:
                available_gpus.append(gpu_instance)
        if len(available_gpus) == 0:
            return -2

        #check gpu memory availability
        for gpu_instance in available_gpus:
            if gpu_instance.get_available_memory() < self.estimated_gpu_memory_per_exp:
                available_gpus.remove(gpu_instance)
        if len(available_gpus) == 0:
            return -3

        # add gpu id to parameters and run command
        gpu_instance = random.choice(available_gpus) # randomly choose one availabel gpu
        command_elements = command.split(" ")
        command_elements = command_elements[:3] + ['--gpu', str(gpu_instance.id)] + command_elements[3:]
        final_command = " ".join(command_elements)
        try:
            fh = open(output_file_path, "w+")
            instance = subprocess.Popen(final_command, shell=True, stdout=fh, stderr=fh)
        except Exception as e:
            return e
        self.running_exps[gpu_instance.id].append([instance, fh])
        return instance.pid

    def refresh(self):
        for gpu_id, gpu_running_exps in self.running_exps.items():
            updated_exps = []
            for i, (instance, fh) in enumerate(gpu_running_exps):
                ret = instance.poll()
                if ret is not None:
                    fh.close()
                    continue
                else:
                    updated_exps.append([instance, fh])
            self.running_exps[gpu_id] = updated_exps

    def running_instance_num(self):
        self.refresh()
        count = 0
        for gpu_id, gpu_running_exps in self.running_exps.items():
            count += len(gpu_running_exps)
        return count

    def kill_all_exps(self):
        for gpu_id, gpu_running_exps in self.running_exps.items():
            for instance, fh in gpu_running_exps:
                instance.kill()
                fh.close()
                print("killed {} on gpu {}".format(instance.pid, gpu_id))
        

class GpuInstance:
    def __init__(self, id, handle):
        self.id = id
        self.running_instances = []
        self.handle = handle

    
    def get_available_memory(self):
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        free_mem = meminfo.free / 1024 / 1024
        return free_mem
    