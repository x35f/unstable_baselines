import numpy as np
from numpy.core.numeric import roll
import torch

def soft_update_network(source_network, target_network, tau):
    for target_param, local_param in zip(target_network.parameters(),
                                        source_network.parameters()):
        target_param.data.copy_(tau*local_param.data + (1-tau)*target_param.data)

def dict_batch_generator(data, batch_size, keys=None):
    if keys is None:
        keys = list(data.keys())
    num_data = len(data[keys[0]])
    num_batches = int(np.ceil(num_data / batch_size))
    indices = np.arange(num_data)
    np.random.shuffle(indices)
    for batch_id in range(num_batches):
        batch_start = batch_id * batch_size
        batch_end = min(num_data, (batch_id + 1) * batch_size)
        batch_data = {}
        for key in keys:
            batch_data[key] = data[key][indices[batch_start:batch_end]]
        yield batch_data

def minibatch_rollout(data, rollout_fn, batch_size = 256):
    data_size = len(data)
    num_batches = int(np.ceil(data_size/batch_size))
    results = []

    for i in range(num_batches):
        batch_start = i * batch_size
        batch_end = min(data_size, (i + 1) * batch_size)
        output = rollout_fn(data[batch_start:batch_end])
        if i == 0:
            if isinstance(output, tuple):
                multi_op = True
                results = [[] for _ in output]
            else:
                multi_op = False
        if multi_op:
            for i, op in enumerate(output):
                results[i].append(op)
        else:
            results.append(output)
    if multi_op:
        results = [torch.stack(re) for re in results]
    return results 


def merge_data_batch(data1_dict, data2_dict):
    for key in data1_dict:
        if isinstance(data1_dict[key], np.ndarray):
            data1_dict[key]  = np.concatenate([data1_dict[key], data2_dict[key]], axis=0)
        elif isinstance(data1_dict[key], torch.Tensor):
            data1_dict[key]  = torch.cat([data1_dict[key], data2_dict[key]], dim=0)
    return data1_dict