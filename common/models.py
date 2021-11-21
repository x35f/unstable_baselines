#implement model to learn state transitions and rewards
import torch
from common.networks import get_network, get_act_cls
import torch.nn as nn
from abc import abstractmethod
import numpy as np
import torch.nn.functional as F
from common import util 


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
    def __init__(self,obs_dim, action_dim, hidden_dims, act_fn="relu", out_act_fn="identity", **kwargs):
        super(BaseModel, self).__init__()
        if type(hidden_dims) == int:
            hidden_dims = [hidden_dims]
        hidden_dims = [obs_dim + action_dim] + hidden_dims 
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.networks = []
        act_cls = get_act_cls(act_fn)
        out_act_cls = get_act_cls(out_act_fn)
        for i in range(len(hidden_dims)-1):
            curr_shape, next_shape = hidden_dims[i], hidden_dims[i+1]
            curr_network = get_network([curr_shape, next_shape])
            self.networks.extend([curr_network, act_cls()])
        self.output_dim = obs_dim + 1
        final_network = get_network([hidden_dims[-1], self.output_dim * 2])
        self.networks.extend([final_network, out_act_cls()])
        self.networks = nn.ModuleList(self.networks)

        self.max_logvar = nn.Parameter((torch.ones((1, self.output_dim)).float() / 2), requires_grad=False)
        self.min_logvar = nn.Parameter((-torch.ones((1, self.output_dim)).float() * 10), requires_grad=False)
    
    def forward(self, state, action):
        out = torch.cat([state, action], 1)
        for i, layer in enumerate(self.networks):
            out = layer(out)

        mean = out[:, :self.output_dim]

        logvar = self.max_logvar - F.softplus(self.max_logvar - out[:, self.output_dim:])
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        return mean, torch.exp(logvar)
    
    def predict(self, state, action):
        pass


class EnsembleModel(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dims, num_models = 10, num_elite=5, act_fn="relu", out_act_fn="identity", **kwargs):
        super(EnsembleModel, self).__init__()
        self.models = [BaseModel(obs_dim, action_dim, hidden_dims, act_fn=act_fn, out_act_fn=out_act_fn) for _ in range(num_models)]
        self.models = nn.ModuleList(self.models)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_elite = num_elite
        self.elite_model_idxes = torch.tensor([i for i in range(num_elite)])

    def predict(self, state, action):
        # model_input = torch.cat([state, action], 1)
        if type(state) != torch.Tensor:
            if len(state.shape) == 1:
                state = torch.FloatTensor([state]).to(util.device)
                action = torch.FloatTensor([action]).to(util.device)
            else:
                state = torch.FloatTensor(state).to(util.device)
                action = torch.FloatTensor(action).to(util.device)
        predictions = [model(state, action) for model in self.models]
        return predictions


class PredictEnv:
    def __init__(self, model, env_name):
        self.model = model
        self.env_name = env_name

    def _termination_fn(self, env_name, obs, act, next_obs):
        # TODO
        if env_name == "Hopper-v2":
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

            height = next_obs[:, 0]
            angle = next_obs[:, 1]
            not_done = np.isfinite(next_obs).all(axis=-1) \
                       * np.abs(next_obs[:, 1:] < 100).all(axis=-1) \
                       * (height > .7) \
                       * (np.abs(angle) < .2)

            done = ~not_done
            done = done[:, None]
            return done
        elif env_name == "Walker2d-v2":
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

            height = next_obs[:, 0]
            angle = next_obs[:, 1]
            not_done = (height > 0.8) \
                       * (height < 2.0) \
                       * (angle > -1.0) \
                       * (angle < 1.0)
            done = ~not_done
            done = done[:, None]
            return done
        elif 'walker_' in env_name:
            torso_height =  next_obs[:, -2]
            torso_ang = next_obs[:, -1]
            if 'walker_7' in env_name or 'walker_5' in env_name:
                offset = 0.
            else:
                offset = 0.26
            not_done = (torso_height > 0.8 - offset) \
                       * (torso_height < 2.0 - offset) \
                       * (torso_ang > -1.0) \
                       * (torso_ang < 1.0)
            done = ~not_done
            done = done[:, None]
            return done

    def _get_logprob(self, x, means, variances):

        k = x.shape[-1]

        ## [ num_networks, batch_size ]
        log_prob = -1 / 2 * (k * np.log(2 * np.pi) + np.log(variances).sum(-1) + (np.power(x - means, 2) / variances).sum(-1))

        ## [ batch_size ]
        prob = np.exp(log_prob).sum(0)

        ## [ batch_size ]
        log_prob = np.log(prob)

        stds = np.std(means, 0).mean(-1)

        return log_prob, stds

    def step(self, obs, act, deterministic=False):
        if len(obs.shape) == 1:
            obs = obs[None]
            act = act[None]
            return_single = True
        else:
            return_single = False

        predictions = self.model.predict(obs, act)
        ensemble_model_means, ensemble_model_vars = [], []
        for (mean, var) in predictions:
            ensemble_model_means.append(mean.detach().cpu().numpy())
            ensemble_model_vars.append(var.detach().cpu().numpy())
        ensemble_model_means, ensemble_model_vars = \
            np.array(ensemble_model_means), np.array(ensemble_model_vars)
        ensemble_model_means[:, :, :-1] += obs
        ensemble_model_stds = np.sqrt(ensemble_model_vars)

        if deterministic:
            ensemble_samples = ensemble_model_means
        else:
            ensemble_samples = ensemble_model_means + np.random.normal(size=ensemble_model_means.shape) * ensemble_model_stds

        num_models, batch_size, _ = ensemble_model_means.shape

        model_idxes = np.random.choice(self.model.elite_model_idxes, size=batch_size)
        batch_idxes = np.arange(0, batch_size)

        samples = ensemble_samples[model_idxes, batch_idxes]
        model_means = ensemble_model_means[model_idxes, batch_idxes]
        model_stds = ensemble_model_stds[model_idxes, batch_idxes]

        log_prob, dev = self._get_logprob(samples, ensemble_model_means, ensemble_model_vars)

        next_obs, rewards = samples[:, :-1] + obs, samples[:, -1]
        terminals = self._termination_fn(self.env_name, obs, act, next_obs)

        batch_size = model_means.shape[0]
        return_means = np.concatenate((model_means[:, :-1], terminals, model_means[:, -1:]), axis=-1)
        return_stds = np.concatenate((model_stds[:, :-1], np.zeros((batch_size, 1)), model_stds[:, -1:]), axis=-1)

        if return_single:
            next_obs = next_obs[0]
            return_means = return_means[0]
            return_stds = return_stds[0]
            rewards = rewards[0]
            terminals = terminals[0]

        assert(type(next_obs) == np.ndarray)

        info = {'mean': return_means, 'std': return_stds, 'log_prob': log_prob, 'dev': dev}
        return next_obs, rewards, terminals, info
