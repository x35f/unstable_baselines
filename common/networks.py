import torch
import torch.nn as nn
class value_network(nn.Module):
    def __init__(self,num_inputs, num_actions, hidden_dims):
        if type(hidden_dims) == int:
            hidden_dims = [hidden_dims]

class policy_network(nn.Module):
    def __init__(self,num_inputs, num_actions, hidden_dims):
        if type(hidden_dims) == int:
            hidden_dims = [hidden_dims]
        