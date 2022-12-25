import warnings
from abc import ABC, abstractmethod
from typing import List, Tuple, Any, Sequence, Union, final

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal
import gym
from gym.spaces import Discrete, Box, MultiBinary, space

from unstable_baselines.common import util
import torch.nn.functional as F
import warnings


def get_optimizer(optimizer_class: str, network: nn.Module, learning_rate: float, **kwargs):
    """
    Parameters
    ----------
    optimizer_class: ['adam', 'sgd'], optional
        The optimizer class.

    network: torch.nn.Module
        The network selected to optimize.

    learning_rate: float

    Return
    ------
    """
    optimizer_fn = optimizer_class.lower()
    if optimizer_fn == "adam":
        optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    elif optimizer_fn == "sgd":
        optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate)
    else:
        raise NotImplementedError(f"Unimplemented optimizer {optimizer_class}.")
    return optimizer


def get_network(in_shape, net_param):
    """
    Parameters
    ----------
    in_shape:
        type: int or tuple
    net_param 
        type: tuple
        format: ("net type", *net_parameters)
    """
    (net_type, *net_args) = net_param
    if isinstance(in_shape, tuple) and len(in_shape) == 1:
        in_shape = in_shape[0] 
    if net_type == 'mlp':
        assert isinstance(in_shape, int) and len(net_args) == 1
        out_shape = net_args[0]    
        net = torch.nn.Linear(in_shape, out_shape)
    elif net_type == 'conv2d':
        assert isinstance(in_shape, tuple) and len(in_shape) == 3 and len(net_args) == 4
        out_channel, kernel_size, stride, padding = net_args
        assert padding >= 0 and stride >= 1 and kernel_size > 0
        in_channel, h, w = in_shape
        net = torch.nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        out_h = int((h + 2 * padding - 1 * (kernel_size - 1) - 1 ) / stride + 1)
        out_w = int((w + 2 * padding - 1 * (kernel_size - 1) - 1 ) / stride + 1)
        out_shape = (out_channel, out_h, out_w)
    elif net_type == "flatten":
        assert isinstance(in_shape, tuple) and len(in_shape) == 3
        net = torch.nn.Flatten()
        out_shape = int(np.prod(in_shape))
    elif net_type in ['maxpool2d', 'avgpool2d']:
        kernel_size, stride, padding = net_param
        assert padding >= 0 and stride >= 1 and kernel_size > 0 and len(in_shape) == 3
        c, h, w = in_shape
        if net_type == "maxpool2d":
            net = torch.nn.MaxPool2d(kernel_size, stride, padding)
        elif net_type == "avgpool2d":
            net = torch.nn.AvgPool2d(kernel_size, stride, padding)
        else:
            raise NotImplementedError
        out_h = int((h + 2 * padding - 1 * (kernel_size - 1) - 1 ) / stride + 1)
        out_w = int((w + 2 * padding - 1 * (kernel_size - 1) - 1 ) / stride + 1)
        out_shape = (c, out_h, out_w)
    else:
        raise ValueError(f"Network params {net_param} illegal.")

    return net, out_shape

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        x = x * torch.sigmoid(x)
        return x


def get_act_cls(act_fn_name):
    act_fn_name = act_fn_name.lower()
    if act_fn_name == "tanh":
        act_cls = torch.nn.Tanh
    elif act_fn_name == "sigmoid":
        act_cls = torch.nn.Sigmoid
    elif act_fn_name == 'relu':
        act_cls = torch.nn.ReLU
    elif act_fn_name == 'leakyrelu':
        act_cls = torch.nn.LeakyReLU
    elif act_fn_name == 'identity':
        act_cls = torch.nn.Identity
    elif act_fn_name == 'swish':
        act_cls = Swish
    else:
        raise NotImplementedError(f"Activation functtion {act_fn_name} is not implemented. \
            Possible choice: ['tanh', 'sigmoid', 'relu', 'identity'].")
    return act_cls

class SequentialNetwork(nn.Module):

    def __init__(
            self, in_shape: int,
            out_shape: int,
            network_params: list,
            act_fn="relu",
            out_act_fn="identity",
            **kwargs
    ):
        super(SequentialNetwork, self).__init__()
        if len(kwargs.keys()) > 0:
            warn_str = "Redundant parameters for SequentialNetwork {}.".format(kwargs)
            warnings.warn(warn_str)
        ''' network parameters:
            int: mlp hidden dim
            str: different kinds of pooling
            (in_channel, out_channel, stride, padding): conv2d
        ''' 
        self.networks = []
        curr_shape = in_shape
        if isinstance(act_fn, str):
            act_cls = get_act_cls(act_fn)
            act_cls_list = [act_cls for _ in network_params]
        else:
            act_cls_list = [get_act_cls(act_f) for act_f in act_fn]

        out_act_cls = get_act_cls(out_act_fn)

        for i, (net_param, act_cls) in enumerate(zip(network_params, act_cls_list)):
            curr_network, curr_shape = get_network(curr_shape, net_param)
            self.networks.extend([curr_network, act_cls()])

        #final network only support mlp
        final_net_params = ('mlp', out_shape)
        final_network, final_shape = get_network(curr_shape, final_net_params)

        self.networks.extend([final_network, out_act_cls()])
        
        self.networks = nn.Sequential(*self.networks)

    def forward(self, input):
        return self.networks(input)

    @property
    def weights(self):
        return [net.weight for net in self.networks if isinstance(net, torch.nn.modules.linear.Linear) or isinstance(net, torch.nn.modules.Conv2d)]

def get_old_network(param_shape, deconv=False):
    """
    Parameters
    ----------
    param_shape: tuple, length:[(4, ), (2, )], optional
    deconv: boolean
        Only work when len(param_shape) == 4. 
    """

    if len(param_shape) == 4:
        if deconv:
            in_channel, kernel_size, stride, out_channel = param_shape
            return torch.nn.ConvTranspose2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride)
        else:
            in_channel, kernel_size, stride, out_channel = param_shape
            return torch.nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride)
    elif len(param_shape) == 2:
        in_dim, out_dim = param_shape
        return torch.nn.Linear(in_dim, out_dim)
    else:
        raise ValueError(f"Network shape {param_shape} illegal.")
    
class MLPNetwork(nn.Module):

    def __init__(
            self, input_dim: int,
            out_dim: int,
            hidden_dims: Union[int, list],
            act_fn="relu",
            out_act_fn="identity",
            **kwargs
    ):
        super(MLPNetwork, self).__init__()
        if len(kwargs.keys()) > 0:
            warn_str = "Redundant parameters for MLP network {}.".format(kwargs)
            warnings.warn(warn_str)

        if type(hidden_dims) == int:
            hidden_dims = [hidden_dims]
        hidden_dims = [input_dim] + hidden_dims
        self.networks = []
        act_cls = get_act_cls(act_fn)
        out_act_cls = get_act_cls(out_act_fn)

        for i in range(len(hidden_dims) - 1):
            curr_shape, next_shape = hidden_dims[i], hidden_dims[i + 1]
            curr_network = get_old_network([curr_shape, next_shape])
            self.networks.extend([curr_network, act_cls()])
        final_network = get_old_network([hidden_dims[-1], out_dim])

        self.networks.extend([final_network, out_act_cls()])
        self.networks = nn.Sequential(*self.networks)

    def forward(self, input):
        return self.networks(input)

    @property
    def weights(self):
        return [net.weight for net in self.networks if isinstance(net, torch.nn.modules.linear.Linear)]


class BasePolicyNetwork(ABC, nn.Module):
    def __init__(self,
                 observation_space: Union[gym.spaces.box.Box, gym.spaces.discrete.Discrete],
                 action_space: gym.Space,
                 network_params: Union[Sequence[tuple], tuple],
                 act_fn: str = "relu",
                 *args, **kwargs
                 ):
        super(BasePolicyNetwork, self).__init__()

        self.observation_space = observation_space
        self.action_space = action_space
        self.args = args
        self.kwargs = kwargs

        # if isinstance(hidden_dims, int):
        #     hidden_dims = [hidden_dims]
        # hidden_dims = [input_dim] + hidden_dims

        # # init hidden layers
        # self.hidden_layers = []
        # act_cls = get_act_cls(act_fn)
        # for i in range(len(hidden_dims) - 1):
        #     curr_shape, next_shape = hidden_dims[i], hidden_dims[i + 1]
        #     curr_network = get_network([curr_shape, next_shape])
        #     self.hidden_layers.extend([curr_network, act_cls()])

        # init output layer shape
        if isinstance(action_space, Discrete):
            self.action_dim = action_space.n
        elif isinstance(action_space, Box):
            self.action_dim = action_space.shape[0]
        elif isinstance(action_space, MultiBinary):
            self.action_dim = action_space.shape[0]
        else:
            raise TypeError

    @abstractmethod
    def forward(self, state):
        raise NotImplementedError

    @abstractmethod
    def sample(self, state, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def evaluate_actions(self, states, actions, *args, **kwargs):
        raise NotImplementedError

    def to(self, device):
        return nn.Module.to(self, device)


class DeterministicPolicyNetwork(BasePolicyNetwork):
    def __init__(self,
                 observation_space: Union[gym.spaces.box.Box, gym.spaces.discrete.Discrete],
                 action_space: gym.Space,
                 network_params: Union[Sequence[tuple], tuple],
                 act_fn: str = "relu",
                 out_act_fn: str = "identity",
                 *args, **kwargs
                 ):
        super(DeterministicPolicyNetwork, self).__init__(observation_space, action_space, network_params, act_fn)

        self.deterministic = True
        self.policy_type = "deterministic"

        # get final layer
        # final_network = get_network([hidden_dims[-1], self.action_dim])
        # out_act_cls = get_act_cls(out_act_fn)
        # self.networks = nn.Sequential(*self.hidden_layers, final_network, out_act_cls())

        self.networks = SequentialNetwork(observation_space.shape, action_space.shape[0], network_params, act_fn, out_act_fn)


        # set noise
        self.noise = torch.Tensor(self.action_dim)

        # set scaler
        if action_space is None:
            self.register_buffer("action_scale", torch.tensor(1., dtype=torch.float, device=util.device))
            self.register_buffer("action_bias", torch.tensor(0., dtype=torch.float, device=util.device))
        elif not isinstance(action_space, Discrete):
            self.register_buffer("action_scale",
                                 torch.tensor((action_space.high - action_space.low) / 2.0, dtype=torch.float,
                                              device=util.device))
            self.register_buffer("action_bias",
                                 torch.tensor((action_space.high + action_space.low) / 2.0, dtype=torch.float,
                                              device=util.device))

    def forward(self, state: torch.Tensor):
        out = self.networks(state)
        return out

    def sample(self, state: torch.Tensor):
        action_prev_tanh = self.networks(state)
        action_raw = torch.tanh(action_prev_tanh)
        action_scaled = action_raw * self.action_scale + self.action_bias

        return {
            "action_prev_tanh": action_prev_tanh,
            "action_raw": action_raw,
            "action_scaled": action_scaled,
        }

    # CHECK: I'm not sure about the reparameterization trick used in DDPG
    def evaluate_actions(self, state: torch.Tensor):
        action_prev_tanh = self.networks(state)
        action_raw = torch.tanh(action_prev_tanh)
        action_scaled = action_raw * self.action_scale + self.action_bias

        return {
            "action_prev_tanh": action_prev_tanh,
            "action_raw": action_raw,
            "action_scaled": action_scaled,
        }

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(DeterministicPolicyNetwork, self).to(device)


class CategoricalPolicyNetwork(BasePolicyNetwork):
    def __init__(self,
                 input_dim: int,
                 action_space: gym.Space,
                 hidden_dims: Union[Sequence[int], int],
                 act_fn: str = "relu",
                 out_act_fn: str = "identity",
                 *args, **kwargs
                 ):
        super(CategoricalPolicyNetwork, self).__init__(input_dim, action_space, hidden_dims, act_fn, *args, **kwargs)

        self.determnistic = False
        self.policy_type = "categorical"

        # get final layer
        final_network = get_network([hidden_dims[-1], self.action_dim])
        out_act_cls = get_act_cls(out_act_fn)
        self.networks = nn.Sequential(*self.hidden_layers, final_network, out_act_cls())

        # categorical do not have scaler, and do not support re_parameterization

    def forward(self, state: torch.Tensor):
        out = self.networks(state)
        return out

    def sample(self, state: torch.Tensor, deterministic=False):
        logit = self.forward(state)
        probs = torch.softmax(logit, dim=-1)
        if deterministic:
            return {
                "logit": logit,
                "probs": probs,
                "action": torch.argmax(probs, dim=-1, keepdim=True),
                "log_prob": torch.log(torch.max(probs, dim=-1, keepdim=True) + 1e-6),
            }
        else:
            dist = Categorical(probs=probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return {
                "logit": logit,
                "probs": probs,
                "action": action.view(-1, 1),
                "log_prob": log_prob.view(-1, 1),
            }

    def evaluate_actions(self, states, actions, *args, **kwargs):
        if len(actions.shape) == 2:
            actions = actions.view(-1)
        logit = self.forward(states)
        probs = torch.softmax(logit, dim=1)
        dist = Categorical(probs)
        return dist.log_prob(actions).view(-1, 1), dist.entropy().view(-1, 1)

    def to(self, device):
        return super(CategoricalPolicyNetwork, self).to(device)


class GaussianPolicyNetwork(BasePolicyNetwork):
    def __init__(self,
                 observation_space: Union[gym.spaces.box.Box, gym.spaces.discrete.Discrete],
                 action_space: gym.Space,
                 network_params: Union[Sequence[tuple], tuple],
                 act_fn: str = "relu",
                 out_act_fn: str = "identity",
                 re_parameterize: bool = True,
                 predicted_std: bool = True,
                 parameterized_std: bool = False,
                 log_std: float = None,
                 log_std_min: int = -20,
                 log_std_max: int = 2,
                 stablize_log_prob: bool = True,
                 **kwargs
                 ):
        super(GaussianPolicyNetwork, self).__init__(observation_space, action_space, network_params, act_fn)

        self.deterministic = False
        self.policy_type = "Gaussian"
        self.predicted_std = predicted_std
        self.re_parameterize = re_parameterize
        if self.predicted_std:
            self.networks = SequentialNetwork(observation_space.shape, action_space.shape[0] * 2, network_params, act_fn, out_act_fn)
        else:
            self.networks = SequentialNetwork(observation_space.shape, action_space.shape[0], network_params, act_fn, out_act_fn)


        # set scaler
        if action_space is None:
            self.register_buffer("action_scale", torch.tensor(1., dtype=torch.float, device=util.device))
            self.register_buffer("action_bias", torch.tensor(0., dtype=torch.float, device=util.device))
        elif not isinstance(action_space, Discrete):
            self.register_buffer("action_scale",
                                 torch.tensor((action_space.high - action_space.low) / 2.0, dtype=torch.float,
                                              device=util.device))
            self.register_buffer("action_bias",
                                 torch.tensor((action_space.high + action_space.low) / 2.0, dtype=torch.float,
                                              device=util.device))

        # set log_std
        if log_std == None:
            self.log_std = -0.5 * np.ones(self.action_dim, dtype=np.float32)
        else:
            self.log_std = log_std
        if parameterized_std:
            self.log_std = torch.nn.Parameter(torch.as_tensor(self.log_std)).to(util.device)
        else:
            self.log_std = torch.tensor(self.log_std, dtype=torch.float, device=util.device)

        self.register_buffer("log_std_min", torch.tensor(log_std_min, dtype=torch.float, device=util.device))
        self.register_buffer("log_std_max", torch.tensor(log_std_max, dtype=torch.float, device=util.device))
        self.stablize_log_prob = stablize_log_prob

    def forward(self, obs: torch.Tensor):
        out = self.networks(obs)
        action_mean = out[:, :self.action_dim]
        # check whether the `log_std` is fixed in forward() to make the sample function
        # keep consistent
        if self.predicted_std:
            action_log_std = out[:, self.action_dim:]    
        else:   
            action_log_std = self.log_std
        return action_mean, action_log_std

    def sample(self, obs: torch.Tensor, deterministic: bool = False):

        mean, log_std = self.forward(obs)
        # util.debug_print(type(log_std), info="Gaussian Policy sample")
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max).expand_as(mean)

        dist = Normal(mean, log_std.exp())
        if deterministic:
            # action sample must be detached !
            action_prev_tanh = mean.detach()
        else:
            if self.re_parameterize:
                action_prev_tanh = dist.rsample()
            else:
                action_prev_tanh = dist.sample()

        action_raw = torch.tanh(action_prev_tanh)
        action_scaled = action_raw * self.action_scale + self.action_bias

        log_prob_prev_tanh = dist.log_prob(action_prev_tanh)
        # log_prob = log_prob_prev_tanh - torch.log(self.action_scale*(1-torch.tanh(action_prev_tanh).pow(2)) + 1e-6)
        if self.stablize_log_prob:
            log_prob = log_prob_prev_tanh - (
                    2 * (np.log(2) - action_prev_tanh - torch.nn.functional.softplus(-2 * action_prev_tanh)))
        else:
            log_prob = log_prob_prev_tanh
        log_prob = torch.sum(log_prob, dim=-1, keepdim=True)
        return {
            "action_mean_raw": mean,
            "action_scaled": action_scaled, 
            "log_prob": log_prob,
            "log_std": log_std
        }

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor, action_type: str = "scaled"):
        """ Evaluate action to get log_prob and entropy.
        
        Note: This function should not be used by SAC because SAC only replay states in buffer.
        """
        
        if torch.any(torch.isnan(obs)):
            print("obs nan", obs)
        mean, log_std = self.forward(obs)
        if torch.any(torch.isnan(mean)):
            print("mean nan", mean)
        if torch.any(torch.isnan(log_std)):
            print("log_std nan", log_std)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max).expand_as(mean)
        if torch.any(torch.isnan(log_std)):
            print("after log_std nan", log_std)
        dist = Normal(mean, log_std.exp())

        if action_type == "scaled":
            actions = (actions - self.action_bias) / self.action_scale
            actions = torch.atanh(actions)
        elif action_type == "raw":
            # actions = torch.atanh(actions)
            pass
        log_prob = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        
        if torch.any(torch.isnan(log_prob)):
            print("log_prob nan")
        if torch.any(torch.isnan(entropy)):
            print("entropy nan")
        return {
            "log_prob": log_prob,
            "entropy": entropy
        }

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicyNetwork, self).to(device)


class PolicyNetworkFactory():
    @staticmethod
    def get(
            observation_space: Union[gym.spaces.box.Box, gym.spaces.discrete.Discrete],
            action_space: gym.Space,
            network_params: Union[Sequence[int], int],
            act_fn: str = "relu",
            out_act_fn: str = "identity",
            deterministic: bool = False,
            distribution_type: str = None,
            *args, **kwargs
    ):
        cls = None
        if deterministic:
            cls = DeterministicPolicyNetwork
        elif not distribution_type is None:
            cls = {
                "deterministic": DeterministicPolicyNetwork,
                "gaussian": GaussianPolicyNetwork,
                "categorical": CategoricalPolicyNetwork
            }.get(distribution_type)
        elif isinstance(action_space, Discrete):
            cls = CategoricalPolicyNetwork
        elif isinstance(action_space, Box):
            cls = GaussianPolicyNetwork
        else:
            raise ArithmeticError(
                f"Cannot determine policy network type from arguments - deterministic: {deterministic}, distribution_type: {distribution_type}, action_space: {action_space}.")
        return cls(observation_space, action_space, network_params, act_fn, out_act_fn, *args,
                   **kwargs)
