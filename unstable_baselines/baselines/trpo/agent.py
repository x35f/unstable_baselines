from operator import itemgetter
import torch
import torch.nn.functional as F
import math
import numpy as np
import scipy
from torch.autograd import Variable
from unstable_baselines.common.agents import BaseAgent
from unstable_baselines.common.networks import MLPNetwork, PolicyNetworkFactory
from unstable_baselines.common import util 
from unstable_baselines.common.functional import set_flattened_params, get_flattened_params, get_flat_grads

class TRPOAgent(BaseAgent):
    def __init__(self,observation_space, action_space,
           l2_reg,
           v_optimize_maxiter, # 25
           num_conjugate_gradient_steps, # 10
           damping, # 0.1
           max_kl_div, # 1e-2
           max_backtracks, # 10
           accept_ratio, # 0.1
           residual_tol, # 1e-10
           gamma,
           tau,
            **kwargs):
        super(TRPOAgent, self).__init__()
        obs_dim = observation_space.shape[0]
        
        #initilze networks
        self.v_network = MLPNetwork(obs_dim, 1, **kwargs['v_network'])
        self.policy_network = PolicyNetworkFactory.get(obs_dim, action_space,  **kwargs['policy_network'])
        #self.policy_network = Policy(obs_dim, action_space.shape[0])

        #pass to util.device
        self.v_network = self.v_network.to(util.device)
        self.policy_network = self.policy_network.to(util.device)

        #register networks
        self.networks = {
            'v_network': self.v_network,
            'policy_network': self.policy_network
        }


        #hyper-parameters
        self.l2_reg = l2_reg
        self.v_optimize_maxiter = v_optimize_maxiter
        self.num_conjugate_gradient_steps = num_conjugate_gradient_steps
        self.damping = damping
        self.max_kl_div = max_kl_div
        self.max_backtracks = max_backtracks
        self.accept_ratio = accept_ratio
        self.residual_tol = residual_tol
        self.gamma = gamma
        self.tau = tau


    def update(self, data_batch):
        self.log_info = {}
        obs_batch, action_batch, reward_batch, done_batch = \
            itemgetter("obs", "action", "reward","done")(data_batch)
        reward_batch = reward_batch.squeeze(1)
        done_batch = done_batch.squeeze(1)
        with torch.no_grad():
            value_batch = self.v_network(obs_batch)

        return_batch = torch.FloatTensor(action_batch.shape[0], 1).to(util.device)
        deltas = torch.FloatTensor(action_batch.shape[0], 1).to(util.device)
        advantage_batch = torch.FloatTensor(action_batch.shape[0], 1).to(util.device)

        #compute return and advantage batch
        prev_return = 0
        prev_value = 0
        prev_advantage = 0
        for i in reversed(range(reward_batch.size(0))):
            return_batch[i] = reward_batch[i] + self.gamma * prev_return * (1.0 - done_batch[i])
            deltas[i] = reward_batch[i] + self.gamma * prev_value * (1.0 - done_batch[i]) - value_batch.data[i]
            advantage_batch[i] = deltas[i] + self.gamma * self.tau * prev_advantage * (1.0 - done_batch[i])

            prev_return = return_batch[i, 0]
            prev_value = value_batch.data[i, 0]
            prev_advantage = advantage_batch[i, 0]

        #normalize advantage
        advantage_batch =  (advantage_batch - advantage_batch.mean()) / advantage_batch.std()

        # optimize value net
        self.optimize_value_net(obs_batch, return_batch)

        #optimize policy network
        with torch.no_grad():
            action_mean, action_log_std = itemgetter("action_mean_raw", "log_std")(self.policy_network.sample(obs_batch, deterministic=True))
        old_action_mean = action_mean.detach().data
        old_action_log_std = action_log_std.detach().data

        old_log_prob = self.normal_log_density(action_batch, old_action_mean, old_action_log_std).data.clone()
        action_loss = self.compute_action_loss(obs_batch, action_batch, advantage_batch, old_log_prob, volatile=False)

        self.log_info ['loss/action_loss_init'] = action_loss.item()

        action_grads = torch.autograd.grad(action_loss, self.policy_network.parameters())
        loss_grad = torch.cat([grad.view(-1) for grad in action_grads]).data

      
        stepdir = self.conjugate_gradients(-loss_grad, obs_batch, old_action_mean, old_action_log_std)

        shs = 0.5 * (stepdir * self.Fvp(stepdir, obs_batch, old_action_mean, old_action_log_std)).sum(0, keepdim=True)

        lm = torch.sqrt(shs / self.max_kl_div)
        fullstep = stepdir / lm[0]
        #print(shs, lm[0], fullstep)
        neggdotstepdir = (-loss_grad * stepdir).sum(0, keepdim=True)

        self.log_info['misc/lagrange multiplier'] = lm[0]
        self.log_info['misc/grad_norm'] = loss_grad.norm()

        prev_params = get_flattened_params(self.policy_network)
        success, new_params = self.linesearch(obs_batch, action_batch, advantage_batch, old_log_prob, prev_params, fullstep, neggdotstepdir / lm[0])
        #print(success, new_params[:5], prev_params[:5])
        if success:
            set_flattened_params(self.policy_network, new_params)
        return self.log_info

    def normal_log_density(self, x, mean, log_std):
        std = log_std.exp()
        var = std.pow(2)
        log_density = -(x - mean).pow(2) / (
        2 * var) - 0.5 * math.log(2 * math.pi) - log_std
        return log_density.sum(1, keepdim=True)


    def compute_action_loss(self, obs_batch, action_batch, advatange_batch, fixed_log_prob, volatile=False):
        if volatile:
            with torch.no_grad():
                action_mean, action_log_std = itemgetter("action_mean_raw", "log_std")(self.policy_network.sample(obs_batch, deterministic=True))
        else:
            action_mean, action_log_std = itemgetter("action_mean_raw", "log_std")(self.policy_network.sample(obs_batch, deterministic=True))
        log_prob = self.normal_log_density(action_batch, action_mean, action_log_std)
        action_loss = - advatange_batch * torch.exp(log_prob - fixed_log_prob)
        return action_loss.mean()


    def optimize_value_net(self, obs_batch, targets):

        def get_value_loss(flat_params):
            set_flattened_params(self.v_network, torch.Tensor(flat_params))
            for param in self.v_network.parameters():
                if param.grad is not None:
                    param.grad.data.fill_(0)

            values_ = self.v_network(obs_batch)
            value_loss = (values_ - targets).pow(2).mean()
            # weight decay
            for param in self.v_network.parameters():
                value_loss += param.pow(2).sum() * self.l2_reg
            value_loss.backward()
            return (value_loss.cpu().data.double().numpy(), get_flat_grads(self.v_network).cpu().data.double().numpy())
        
        flat_params, final_loss, optinfo = scipy.optimize.fmin_l_bfgs_b(get_value_loss, get_flattened_params(self.v_network).cpu().double().numpy(), maxiter=self.v_optimize_maxiter)
        self.log_info['loss/value_loss'] = final_loss
        set_flattened_params(self.v_network, torch.Tensor(flat_params).to(util.device))

    def linesearch(self, obs_batch, action_batch, advantage_batch,
               fixed_log_prob,
               prev_params,
               fullstep,
               expected_improve_rate):
        fval = self.compute_action_loss(obs_batch, action_batch, advantage_batch, fixed_log_prob, volatile=True).data
        self.log_info['misc/fval_before'] = fval.item()
        for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(self.max_backtracks)):
            xnew = prev_params + stepfrac * fullstep
            set_flattened_params(self.policy_network, xnew)
            newfval = self.compute_action_loss(obs_batch, action_batch, advantage_batch, fixed_log_prob, volatile=True).data
            actual_improve = fval - newfval
            expected_improve = expected_improve_rate * stepfrac
            ratio = actual_improve / expected_improve

            if ratio.item() > self.accept_ratio and actual_improve.item() > 0:
                self.log_info['misc/fval_after'] = newfval.item()
                self.log_info['misc/fval_improve'] = actual_improve.item()
                self.log_info['misc/fval_improve_ratio'] = ratio.item()
                return True, xnew
        return False, None


    def Fvp(self, v, obs_batch, fixed_action_mean, fixed_action_log_std):
        #compute kl
        action_mean, action_log_std = itemgetter("action_mean_raw", "log_std")(self.policy_network.sample(obs_batch,deterministic=True))

        action_std = action_log_std.exp()
        action_var = action_std.pow(2)
        #kl_div = action_log_std - fixed_action_log_std + (fixed_action_log_std.exp().pow(2) + (action_mean - fixed_action_mean).pow(2)) / (2.0 * action_var) - 0.5
        kl_div = action_log_std - action_log_std.data + (action_var.data + (action_mean - action_mean.data).pow(2)) / (2.0 * action_var) - 0.5
        kl_div = kl_div.sum(1, keepdim=True).mean()
        grads = torch.autograd.grad(kl_div, self.policy_network.parameters(), create_graph=True)
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

        kl_v = (flat_grad_kl * Variable(v)).sum()
        grads = torch.autograd.grad(kl_v,  self.policy_network.parameters())
        flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).data

        return  v * self.damping + flat_grad_grad_kl

    def conjugate_gradients(self, loss_grad, obs_batch,fixed_action_mean, fixed_action_log_std):
        x = torch.zeros(loss_grad.size()).to(util.device)
        r = loss_grad.clone()
        p = loss_grad.clone()
        rdotr = torch.dot(r, r)
        self.log_info['misc/num_conjugate_gradient_steps'] = self.num_conjugate_gradient_steps
        for i in range(self.num_conjugate_gradient_steps):
            _Avp = self.Fvp(p, obs_batch ,fixed_action_mean, fixed_action_log_std)
            alpha = rdotr / torch.dot(p, _Avp)
            
            x += alpha * p
            r -= alpha * _Avp
            new_rdotr = torch.dot(r, r)
            betta = new_rdotr / rdotr
            p = r + betta * p
            rdotr = new_rdotr
            if rdotr < self.residual_tol:
                self.log_info['misc/num_conjufate_gradient_steps'] = i + 1
                break
        self.log_info['misc/rdotr'] = rdotr.item()
        return x

    @torch.no_grad()
    def select_action(self, obs, deterministic=False):
        if len(obs.shape) == 1:
            ret_single = True
            obs = [obs]
        if type(obs) != torch.tensor:
            obs = torch.FloatTensor(np.array(obs)).to(util.device)
        action, action_mean_raw, log_prob = itemgetter("action_scaled", "action_mean_raw", "log_prob")(self.policy_network.sample(obs, deterministic=deterministic))
        if ret_single:
            action = action[0]
            action_mean_raw = action_mean_raw[0]
            log_prob = log_prob[0]
        return {
            'action': action.detach().cpu().numpy(),
            "action_mean_raw":action_mean_raw.detach().cpu().numpy(),
            'log_prob' : log_prob
            }




