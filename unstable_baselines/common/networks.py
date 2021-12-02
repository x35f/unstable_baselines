from typing_extensions import Required
import torch
import torch.nn as nn
import numpy as np
from unstable_baselines.common import util
def get_optimizer(optimizer_class, network, learning_rate, **kwargs):
    optimizer_fn = optimizer_class.lower()
    if optimizer_fn == "adam":
        optimizer = torch.optim.Adam(network.parameters(),lr = learning_rate)
    elif optimizer_fn== "sgd":
        optimizer = torch.optim.SGD(network.parameters(),lr = learning_rate)
    else:
        assert 0,"Unimplemented optimizer {}".format(optimizer_class)
    return optimizer


def get_network(param_shape, deconv=False):
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
        assert 0, "network parameters {} illegal".format(param_shape)


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
    else:
        assert 0, "activation function {} not implemented".format(act_fn_name)
    return act_cls


class MLPNetwork(nn.Module):
    def __init__(self,input_dim, out_dim, hidden_dims, act_fn="relu", out_act_fn="identity", **kwargs):
        print("redundant parameters for MLP network: {}".format(kwargs))
        super(MLPNetwork, self).__init__()
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
        final_network = get_network([hidden_dims[-1],out_dim])
        self.networks.extend([final_network, out_act_cls()])
        self.networks = nn.Sequential(*self.networks)
    
    def forward(self, input):
        return self.networks(input)



class PolicyNetwork(nn.Module):
    def __init__(self,input_dim, action_space, hidden_dims, act_fn="relu", out_act_fn="identity", deterministic=False, re_parameterize=True,  **kwargs):
        super(PolicyNetwork, self).__init__()
        if type(hidden_dims) == int:
            hidden_dims = [hidden_dims]
        hidden_dims = [input_dim] + hidden_dims 
        if action_space.__class__.__name__ == "Discrete":
            action_dim = action_space.n
            self.dist_cls = torch.distributions.Categorical
            self.policy_type = "discrete"
        elif action_space.__class__.__name__ == "Box":
            action_dim = action_space.shape[0]
            self.dist_cls = torch.distributions.Normal
            self.policy_type = "gaussian"
        elif action_space.__class__.__name__ == "MultiBinary":
            action_dim = action_space.shape[0]
            self.dist_cls = torch.distributions.Bernouli
            self.policy_type = 'MultiBinary'
        else:
            raise NotImplementedError
        self.networks = []
        act_cls = get_act_cls(act_fn)
        out_act_cls = get_act_cls(out_act_fn)
        for i in range(len(hidden_dims)-1):
            curr_shape, next_shape = hidden_dims[i], hidden_dims[i+1]
            curr_network = get_network([curr_shape, next_shape])
            self.networks.extend([curr_network, act_cls()])
        if self.policy_type == "gaussian":
            if re_parameterize:
                #output mean and std for re-parametrization
                final_network = get_network([hidden_dims[-1], action_dim * 2]) 
            else:
                #output mean and let std to be optimizable
                final_network = get_network([hidden_dims[-1], action_dim]) 
                log_std = -0.5 * np.ones(action_dim, dtype=np.float32)
                self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        else:
            raise NotImplementedError
        self.networks.extend([final_network, out_act_cls()])
        self.networks = nn.Sequential(*self.networks)
        #action rescaler
        if action_space is None:
            # set to [0,1] as default
            self.action_scale = torch.tensor(1., dtype=torch.float, requires_grad=False, device=util.device)
            self.action_bias = torch.tensor(0., dtype=torch.float, requires_grad=False, device=util.device)
        else:
            self.action_scale = torch.tensor( (action_space.high - action_space.low) / 2.0, dtype=torch.float, requires_grad=False, device=util.device)
            self.action_bias = torch.tensor( (action_space.high + action_space.low) / 2.0, dtype=torch.float, requires_grad=False, device=util.device)
        self.action_dim = action_dim
        self.noise = torch.Tensor(action_dim) # for deterministic policy
        self.deterministic = deterministic    
        self.re_parameterize = re_parameterize

    def forward(self, state):
        # outs = [None for _ in self.networks]
        # for i, layer in enumerate(self.networks):
        #     if i == 0:
        #         outs[0] = layer(state)
        #     else:
        #         outs[i] = layer(outs[i-1])
        # out = outs[-1]
        out = self.networks(state)
        if self.policy_type == "gaussian":
            if self.re_parameterize:        
                #action_mean, action_log_std = torch.split(out, [self.action_dim, self.action_dim], dim=1)
                action_mean = out[:,:self.action_dim]
                action_log_std = out[:,self.action_dim:]
                if self.deterministic:
                    return action_mean, None
                else:
                    return action_mean, action_log_std
            else:
                return out, None
        else:
            raise NotImplementedError

    def sample(self, state):
        action_mean_raw, action_log_std_raw = self.forward(state)
        if self.deterministic:
            action_mean_scaled = torch.tanh(action_mean_raw) * self.action_scale + self.action_bias
            #noise = self.noise.normal_(0., std=0.1)
            #noise = noise.clamp(-0.25, 0.25)
            #action = action_mean + noise
            action = action_mean_scaled
            return action, torch.tensor(0.), action_mean_scaled
        else:    
            if self.re_parameterize:
                #to reperameterize, use rsample
                action_std_raw = action_log_std_raw.exp()
                dist = self.dist_cls(action_mean_raw, action_std_raw)
                mean_sample_raw = dist.rsample()
                action = torch.tanh(mean_sample_raw) * self.action_scale + self.action_bias
                log_prob_raw = dist.log_prob(mean_sample_raw)
                log_prob_stable = log_prob_raw - torch.log(self.action_scale * (1 - torch.tanh(mean_sample_raw).pow(2)) + 1e-6)
                log_prob = log_prob_stable.sum(1, keepdim=True)
                action_mean_scaled = torch.tanh(action_mean_raw) * self.action_scale + self.action_bias
                return action, log_prob, action_mean_scaled, {
                    "action_std": action_std_raw,
                    "pre_tanh_value": mean_sample_raw # todo: check scale
                    }

            else:
                dist = self.dist_cls(action_mean_raw, torch.exp(self.log_std))
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(axis=-1, keepdim=True)
                return action, log_prob, action_mean_raw


    def evaluate_actions(self, states, actions):
        assert not self.re_parameterize
        action_mean, _ = self.forward(states)
        dist = self.dist_cls(action_mean, torch.exp(self.log_std))
        log_pi = dist.log_prob(actions).sum(-1, keepdim=True)
        ent = dist.entropy().sum(1, keepdim=True)
        return log_pi, ent

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(PolicyNetwork, self).to(device)

        





        