import torch
import torch.nn as nn
class Policy(nn.Module):
    def __init__(self, num_inputs, num_outputs, **args):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(num_inputs, 64)
        self.affine2 = nn.Linear(64, 64)

        self.action_mean = nn.Linear(64, num_outputs)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)

        self.action_log_std = nn.Parameter(torch.zeros(1, num_outputs))

        self.saved_actions = []
        self.rewards = []
        self.final_value = 0

    def forward(self, x):
        x = torch.tanh(self.affine1(x))
        x = torch.tanh(self.affine2(x))

        action_mean = self.action_mean(x)
        action_log_std = self.action_log_std.expand_as(action_mean)

        #action_std = torch.exp(action_log_std)
        return action_mean, action_log_std

    def sample(self, obs, deterministic=False):
        mean, log_std = self.forward(obs)
        if deterministic:
            action = mean
        else:
            action = torch.normal(mean, torch.exp(log_std))
            #print(mean, log_std.exp(), action)
        return {
            "action_scaled": action,
            "action": action,
            "action_mean_raw":mean,
            "log_std":log_std,
            "log_prob":log_std
        } 
