#implement model to learn state transitions and rewards
import torch
from common.networks import get_network, get_act_cls
import torch.nn as nn
class BaseAgent(object):
    def __init__(self,**kwargs):
        super(BaseAgent,self).__init__(**kwargs)
    
    @abstractmethod
    def update(self,data_batch):
        pass

    @abstractmethod
    def select_action(self, state):
        pass

    @abstractmethod
    def load_model(self, dir):
        pass
    
    @abstractmethod
    def save_model(self, target_dir, ite):
        pass

class BaseModel(nn.Module):
    def __init__(self,input_dim, out_dim, hidden_dims, act_fn="relu", out_act_fn="identity", **kwargs):
        super(BaseModel, self).__init__()
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
