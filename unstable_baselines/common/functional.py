import numpy as np
from numpy.core.numeric import roll
import torch
import scipy
from scipy.signal import lfilter

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

def minibatch_inference(args, rollout_fn, batch_size = 1000, cat_dim=0):
    data_size = len(args[0])
    num_batches = int(np.ceil(data_size/batch_size))
    inference_results = []
    for i in range(num_batches):
        batch_start = i * batch_size
        batch_end = min(data_size, (i + 1) * batch_size)
        input_batch = [ip[batch_start:batch_end] for ip in args]
        outputs = rollout_fn(*input_batch)
        if i == 0:
            if isinstance(outputs, tuple):
                multi_op = True
            else:
                multi_op = False
            inference_results = outputs
        else:
            if multi_op:
                inference_results = (torch.cat([prev_re, op], dim=cat_dim) for prev_re, op in zip(inference_results, outputs))
            else:   
                inference_results = torch.cat([inference_results, outputs])         
    return inference_results


def merge_data_batch(data1_dict, data2_dict):
    for key in data1_dict:
        if isinstance(data1_dict[key], np.ndarray):
            data1_dict[key]  = np.concatenate([data1_dict[key], data2_dict[key]], axis=0)
        elif isinstance(data1_dict[key], torch.Tensor):
            data1_dict[key]  = torch.cat([data1_dict[key], data2_dict[key]], dim=0)
    return data1_dict

def discount_cum_sum(x, discount):
    return lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def get_grad_parameters(model):
    for p in model.parameters():
        if p.requires_grad == True:
            yield p
    
def get_flattened_params(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))

    flat_params = torch.cat(params)
    return flat_params

def set_flattened_params(model, params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size

def get_flat_grads(model, grad_grad=False):
    grads = []
    for param in model.parameters():
        if grad_grad:
            grads.append(param.grad.grad.view(-1))
        else:
            grads.append(param.grad.view(-1))

    flat_grad = torch.cat(grads)
    return flat_grad