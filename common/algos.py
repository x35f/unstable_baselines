import numpy as np
import random

def Bernstein_confidence_bound(sigma, b, n):
    return np.sqrt(2 * (sigma ** 2) / n * np.log(2 / sigma)) + \
        2 * b / (3 * n) * np.log(2 / sigma)


def get_bias(bias_estimator, info, **kwargs):
    assert bias_estimator in ['1', '2', '3']
    if bias_estimator == '1':
        buffer = info['buffer']
        max_r = buffer.max_reward
        gamma = buffer.gamma
        H = info['max_trajectory_length']
        n = info['n']
        bias = max_r * (gamma ** n - gamma ** H) / (1 - gamma)
    elif bias_estimator == '2':
        bellman_error = info['bellman_error']
        gamma = info['buffer'].gamma
        n = info['n']
        bias = (gamma ** n) / (1 - gamma ** n) * bellman_error
    elif bias_estimator == '3':
        raise NotImplementedError
    return bias

def get_variance(variance_estimator, info, **kwargs):
    assert variance_estimator in ['1', '2']
    if variance_estimator == "1":
        #estimate max value via max reward
        buffer = info['buffer']
        max_r = buffer.max_reward
        gamma = buffer.gamma
        n = info['n']
        max_v = max_r / gamma
        variance = 3 * ( max_v ** 2) * (1 + (1 - gamma ** (2 * n)) / (1 - gamma ** 2))
    elif variance == "2":
        buffer = info['buffer']
        max_r = buffer.max_reward
        gamma = buffer.gamma
        n = info['n']
        value_network = info['value_network']
        #estimate max v via sampling
        state_batch = buffer.sample_specific_buffer("obs", 10000)
        value = value_network(state_batch).detach().cpu().numpy()
        print(value.shape)
        max_value = np.max(value)
        #in case the value is unreasonably large, bound it by max_r/gamma
        max_value = min(max_r/gamma, max_value)
        variance = 3 * (max_v ** 2) * (1 + (1 - gamma ** (2 * n)) / (1 - gamma ** 2))
    return variance




def calculate_last_intersected(estimates):
    #input: sorted list with elements [n, b, v]
    minn = -np.inf
    maxx = np.inf
    for estimate in estimates:
        bias = estimate['bias']
        n = estimate['n']
        confidence_bound = estimate['confidence_bound']
        curr_min = bias - confidence_bound
        curr_max = bias + confidence_bound
        minn = max(curr_min, minn)
        maxx = min(curr_max, maxx)
        if new_min >= new_max:
            return n
    return estimates[-1]['n']