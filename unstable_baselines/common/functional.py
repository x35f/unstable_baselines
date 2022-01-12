import numpy as np
from numpy.core.numeric import roll
import torch
import scipy

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

def minibatch_inference(args, rollout_fn, batch_size = 1000):
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
                inference_results = (torch.cat([prev_re, op], dim=0) for prev_re, op in zip(inference_results, outputs))
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
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def env_numpy_to_device_tensor( obs):
    """ Call it before need to pass data to networks.

    1. Transform numpy.ndarray to torch.Tensor;
    2. Make sure the tensor have the batch dimension;
    3. Pass the tensor to util.device;
    4. Make sure the type of tensor is float32.
    """
    # util.debug_print(device)
    if not isinstance(obs, torch.Tensor):
        obs = torch.FloatTensor(obs)
        if len(obs.shape) < 2:
            obs = obs.unsqueeze(0)
    obs = obs.to(util.device)
    return obs

def device_tensor_to_env_numpy(self, *args):
    """ Call it before need to pass data cpu.
    """
    return (item.detach().cpu().squeeze().numpy() for item in args)