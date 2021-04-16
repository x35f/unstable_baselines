import torch
import torch.nn as nn
import numpy as np

def get_optimizer(optimizer_fn, network, learning_rate):
    optimizer_fn = optimizer_fn.lower()
    if optimizer_fn == "adam":
        optimizer = torch.optim.Adam(network.parameters(),lr = learning_rate)
    elif optimizer_fn== "sgd":
        optimizer = torch.optim.SGD(network.parameters(),lr = learning_rate)
    else:
        assert 0,"Unimplemented optimizer {}".format(optimizer_fn)
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

class NaiveCNN(nn.Module):
    def __init__(self,input_dim, out_dim, hidden_dims, act_fn="relu", out_act_fn="identity", **kwargs):
        super(QNetwork, self).__init__()
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
        self.networks = nn.ModuleList(self.networks)
    
    def forward(self, state, action):
        out = torch.cat([state, action], 1)
        for i, layer in enumerate(self.networks):
            out = layer(out)
        return out
    
class QNetwork(nn.Module):
    def __init__(self,input_dim, out_dim, hidden_dims, act_fn="relu", out_act_fn="identity", **kwargs):
        super(QNetwork, self).__init__()
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
        self.networks = nn.ModuleList(self.networks)
    
    def forward(self, state, action):
        out = torch.cat([state, action], 1)
        for i, layer in enumerate(self.networks):
            out = layer(out)
        return out

class VNetwork(nn.Module):
    def __init__(self,input_dim, out_dim, hidden_dims, reparameterize=True, act_fn="relu", out_act_fn="identity", **kwargs):
        super(VNetwork, self).__init__()
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
        self.networks = nn.ModuleList(self.networks)
    
    def forward(self, state):
        out = state
        for i, layer in enumerate(self.networks):
            out = layer(out)
        return out


class PolicyNetwork(nn.Module):
    def __init__(self,input_dim, action_space, hidden_dims, act_fn="relu", out_act_fn="identity", deterministic=False, re_parameterize=True, **kwargs):
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
        self.networks = nn.ModuleList(self.networks)
        #action rescaler
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor( (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor( (action_space.high + action_space.low) / 2.)
        self.action_dim = action_dim
        self.noise = torch.Tensor(action_dim) # for deterministic policy
        self.deterministic = deterministic    
        self.re_parameterize = re_parameterize

    def forward(self, state):
        out = state
        for i, layer in enumerate(self.networks):
            out = layer(out)
        if self.policy_type == "gaussian":
            if self.re_parameterize:
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
        action_mean, action_log_std = self.forward(state)
        if self.deterministic:
            action_std = action_log_std.exp()
            action_mean = torch.tanh(action_mean) * self.action_scale + self.action_bias
            noise = self.noise.normal_(0., std=0.1)
            noise = noise.clamp(-0.25, 0.25)
            action = action_mean + noise
            return action, torch.tensor(0.), action_mean
        else:    
            if self.re_parameterize:
                #to reperameterize, use rsample
                action_std = action_log_std.exp()
                dist = self.dist_cls(action_mean, action_std)
                mean_sample = dist.rsample()
                action = torch.tanh(mean_sample) * self.action_scale + self.action_bias
                log_prob = dist.log_prob(mean_sample)
                log_prob -= torch.log(self.action_scale * (1 - torch.tanh(mean_sample).pow(2)) + 1e-6)
                log_prob = log_prob.sum(1, keepdim=True)
                #print(action_mean)
                mean = torch.tanh(action_mean) * self.action_scale + self.action_bias
                return action, log_prob, mean
            else:
                dist = self.dist_cls(action_mean, torch.exp(self.log_std))
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(axis=-1, keepdim=True)
                return action, log_prob, action_mean


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

        





        