import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

import gym
import warnings
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Any, Sequence, Union, final

from gym.spaces import Discrete, Box, MultiBinary, space
from unstable_baselines.common import util
import torch.nn.functional as F


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
        optimizer = torch.optim.SGD(network.parameters(), lr = learning_rate)
    else:
        raise NotImplementedError(f"Unimplemented optimizer {optimizer_class}.")
    return optimizer

def get_network(param_shape, deconv = False):
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
    elif act_fn_name == 'identity':
        act_cls = torch.nn.Identity
    elif act_fn_name == 'swish':
        act_cls = Swish
    else:
        raise NotImplementedError(f"Activation functtion {act_fn_name} is not implemented. \
            Possible choice: ['tanh', 'sigmoid', 'relu', 'identity'].")
    return act_cls


class MLPNetwork(nn.Module):

    def __init__(
            self,input_dim: int, 
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

        for i in range(len(hidden_dims)-1):
            curr_shape, next_shape = hidden_dims[i], hidden_dims[i+1]
            curr_network = get_network([curr_shape, next_shape])
            self.networks.extend([curr_network, act_cls()])
        final_network = get_network([hidden_dims[-1], out_dim])
        self.networks.extend([final_network, out_act_cls()])
        self.networks = nn.Sequential(*self.networks)
    
    def forward(self, input):
        return self.networks(input)
    
    @property 
    def weights(self):
        return [net.weight for net in self.networks if isinstance(net, torch.nn.modules.linear.Linear)]

class EnsembleMLPNetwork(nn.Module):

    def __init__(self, 
            input_dim: int, 
            out_dim: int, 
            hidden_dims: Union[int, list],
            ensemble_size: int,
            act_fn='swish',
            out_act_fn='identity',
            decay_weights = None,
            **kwargs
            ):
        super(EnsembleMLPNetwork, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.ensemble_size = ensemble_size
        self.decay_weights = decay_weights
        self.mlp_networks = [MLPNetwork(input_dim, out_dim, hidden_dims, act_fn=act_fn, out_act_fn=out_act_fn) for _ in range(ensemble_size)]
        self.mlp_networks=nn.ModuleList(self.mlp_networks)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        mlp_outputs = [mlp_network(input) for mlp_network in self.mlp_networks]
        return torch.stack(mlp_outputs)

    def get_decay_loss(self):
        decay_losses = []
        for mlp_net in self.mlp_networks:
            curr_net_decay_losses = [decay_weight * torch.sum(torch.square(weight)) for decay_weight, weight in zip(self.decay_weights,  mlp_net.weights)]
            decay_losses.append(torch.sum(torch.stack(curr_net_decay_losses)))
        return torch.sum(torch.stack(decay_losses))
            

        


class BasePolicyNetwork(ABC, nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 action_space: gym.Space, 
                 hidden_dims: Union[Sequence[int], int], 
                 act_fn: str = "relu", 
                 *args, **kwargs
        ):
        super(BasePolicyNetwork, self).__init__()

        self.input_dim = input_dim
        self.action_space = action_space
        self.args = args
        self.kwargs = kwargs
        
        if isinstance(hidden_dims,  int):
            hidden_dims = [hidden_dims]
        hidden_dims = [input_dim] + hidden_dims

        # init hidden layers
        self.hidden_layers = []
        act_cls = get_act_cls(act_fn)
        for i in range(len(hidden_dims) - 1):
            curr_shape, next_shape = hidden_dims[i], hidden_dims[i+1]
            curr_network = get_network([curr_shape, next_shape])
            self.hidden_layers.extend([curr_network, act_cls()])

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
                 input_dim: int, 
                 action_space: gym.Space, 
                 hidden_dims: Union[Sequence[int], int], 
                 act_fn: str = "relu", 
                 out_act_fn: str = "identity", 
                 *args, **kwargs
        ):
        super(DeterministicPolicyNetwork, self).__init__(input_dim, action_space, hidden_dims, act_fn, *args,  **kwargs)

        self.deterministic = True
        self.policy_type = "deterministic"

        # get final layer
        final_network = get_network([hidden_dims[-1], self.action_dim])
        out_act_cls = get_act_cls(out_act_fn)
        self.networks = nn.Sequential(*self.hidden_layers, final_network, out_act_cls())

        # set noise
        self.noise = torch.Tensor(self.action_dim)

        # set scaler
        if action_space is None:
            self.action_scale = nn.Parameter(torch.tensor(1., dtype=torch.float, device=util.device), requires_grad=False)
            self.action_bias = nn.Parameter(torch.tensor(0., dtype=torch.float, device=util.device), requires_grad=False)
        elif not isinstance(action_space, Discrete):
            self.action_scale = nn.Parameter(torch.tensor( (action_space.high-action_space.low)/2.0, dtype=torch.float, device=util.device), requires_grad=False)
            self.action_bias = nn.Parameter(torch.tensor( (action_space.high+action_space.low)/2.0, dtype=torch.float, device=util.device), requires_grad=False)

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
                 input_dim: int, 
                 action_space: gym.Space, 
                 hidden_dims: Union[Sequence[int], int], 
                 act_fn: str = "relu", 
                 out_act_fn: str = "identity", 
                 re_parameterize: bool = True,
                 log_var_min: int = -20, 
                 log_var_max: int = 2, 
                 *args, **kwargs
        ):
        super(GaussianPolicyNetwork, self).__init__(input_dim, action_space, hidden_dims, act_fn)

        self.deterministic = False
        self.policy_type = "Gaussian"

        # get final layer
        final_network = get_network([hidden_dims[-1], self.action_dim*2])
        out_act_cls = get_act_cls(out_act_fn)
        self.networks = nn.Sequential(*self.hidden_layers, final_network, out_act_cls())

        # set scaler
        if action_space is None:
            self.action_scale = nn.Parameter(torch.tensor(1., dtype=torch.float, device=util.device), requires_grad=False)
            self.action_bias = nn.Parameter(torch.tensor(0., dtype=torch.float, device=util.device), requires_grad=False)
        else:
            self.action_scale = nn.Parameter(torch.tensor( (action_space.high-action_space.low)/2.0, dtype=torch.float, device=util.device), requires_grad=False)
            self.action_bias = nn.Parameter(torch.tensor( (action_space.high+action_space.low)/2.0, dtype=torch.float, device=util.device), requires_grad=False)

        self.re_parameterize = re_parameterize

        self.log_var_min = log_var_min
        self.log_var_max = log_var_max
        self.log_var_min = nn.Parameter(torch.tensor( self.log_var_min, dtype=torch.float, device=util.device ), requires_grad=False)
        self.log_var_max = nn.Parameter(torch.tensor( self.log_var_max, dtype=torch.float, device=util.device ), requires_grad=False)

    def forward(self, state: torch.Tensor):
        out = self.networks(state)
        action_mean = out[:, :self.action_dim]
        action_log_var = out[:, self.action_dim:]
        return action_mean, action_log_var


    def sample(self, state: torch.Tensor, deterministic: bool=False):
        mean, log_var = self.forward(state)
        log_var = torch.clamp(log_var, self.log_var_min, self.log_var_max)

        dist = Normal(mean, log_var.exp())
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
        log_prob = log_prob_prev_tanh - (2 * (np.log(2) - action_prev_tanh - torch.nn.functional.softplus(-2*action_prev_tanh)) )
        log_prob = torch.sum(log_prob, dim=-1, keepdim=True)
        return {
            "action_prev_tanh": action_prev_tanh, 
            "action_raw": action_raw, 
            "action_scaled": action_scaled, 
            "log_prob_prev_tanh": log_prob_prev_tanh.sum(dim=-1, keepdim=True), 
            "log_prob": log_prob
        }

    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor, action_type: str = "scaled"):
        # should not be used by SAC, because SAC only replays states in buffer
        mean, log_var = self.forward(states)
        dist = Normal(mean, log_var.exp())

        if action_type == "scaled":
            actions = (actions - self.action_bias) / self.action_scale
            actions = torch.atanh(actions)
        elif action_type == "raw":
            actions = torch.atanh(actions)
        log_pi = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return log_pi, entropy

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicyNetwork, self).to(device)


class PolicyNetworkFactory():
    @staticmethod
    def get(
            input_dim: int, 
            action_space: gym.Space, 
            hidden_dims: Union[Sequence[int], int], 
            act_fn: str = "relu", 
            out_act_fn: str = "identity", 
            deterministic: bool = False, 
            re_parameterize: bool = True, 
            distribution_type: str = None,
            *args, **kwargs
        ):
        # 工厂方法，为了兼容老版本的代码
        cls = None
        if deterministic:
            cls = DeterministicPolicyNetwork
        elif not distribution_type is None:
            cls = {
                "deterministic": DeterministicPolicyNetwork, 
                "gaussian": GaussianPolicyNetwork, 
                "categorical": CategoricalPolicyNetwork
            } .get(distribution_type)
        elif isinstance(action_space, Discrete):
            cls = CategoricalPolicyNetwork
        elif isinstance(action_space, Box):
            cls = GaussianPolicyNetwork
        else:
            raise ArithmeticError(f"Cannot determine policy network type from arguments - deterministic: {deterministic}, distribution_type: {distribution_type}, action_space: {action_space}.")
        
        return cls(input_dim, action_space, hidden_dims, act_fn, out_act_fn, re_parameterize=re_parameterize, *args, **kwargs)