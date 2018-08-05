from core.trpo import TrpoUpdater
from utils.torchs import *
import scipy
import math


class VariateTrpoUpdater(TrpoUpdater):
    def __init__(self, nets, cfg, use_fim=False):
        self.variate = nets['variate']
        super(VariateTrpoUpdater, self).__init__(nets['policy'], nets['value'], cfg, use_fim=use_fim)

    def get_policy_loss(self):
        log_probs = self.policy_net.get_log_prob(self.states, self.actions)
        prob_ratio = torch.exp(log_probs - self.fixed_log_probs)
        advantage_term = (self.variate(self.states, self.actions) - self.advantages) * prob_ratio
        action_mean, action_log_std = self.policy_net(self.states)
        variate_term = self.variate(self.states, action_mean + action_log_std.exp() * self.xi) * prob_ratio.data
        return (advantage_term - variate_term).mean()

    def __call__(self, batch, log, *args, **kwargs):
        self.states = batch["states"]
        self.actions = batch["actions"]
        self.advantages = batch["advantages"]
        with torch.no_grad():
            self.fixed_log_probs = self.policy_net.get_log_prob(self.states, self.actions).data
            self.mu, self.log_std = self.policy_net(self.states)
            self.xi = (self.actions - self.mu) / self.log_std.exp()

        # update the value networks by L-BFGS
        self.values_targets = batch["value_targets"]
        flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(self.get_value_loss,
                                                                get_flat_params_from(self.value_net).cpu().numpy(),
                                                                maxiter=25)
        set_flat_params_to(self.value_net, np_to_tensor(flat_params))
        value_loss = (self.value_net(self.states) - self.values_targets).pow(2).mean()
        log["value loss"] = value_loss.item()

        # update the policy networks by trust region gradient
        policy_loss = self.get_policy_loss()
        log["policy loss"] = policy_loss.item()
        grads = torch.autograd.grad(policy_loss, self.policy_net.parameters())
        loss_grad = torch.cat([grad.view(-1) for grad in grads]).data
        stepdir = self.conjugate_gradients(-loss_grad)

        shs = 0.5 * (stepdir.dot(self.Fvp(stepdir)))
        lm = math.sqrt(self.max_kl / shs)
        fullstep = stepdir * lm
        expected_improve = -loss_grad.dot(fullstep)
        prev_params = get_flat_params_from(self.policy_net)
        success, new_params = self.line_search(prev_params, fullstep, expected_improve)
        set_flat_params_to(self.policy_net, new_params)

        return log


class VariateUpdater(object):
    def __init__(self, nets, optimizers, cfg):
        self.policy = nets['policy']
        self.value = nets['value']
        self.variate = nets['variate']
        self.optimizer_policy = optimizers['optimizer_policy']
        self.optimizer_value = optimizers['optimizer_value']
        self.optimizer_variate = optimizers['optimizer_variate']
        self.optim_variate_iterval = cfg["opt_iterval"]
        self.optim_variate_iternum = cfg['optim_variate_iternum'] if 'optim_variate_iternum' in cfg else 1
        self.gpu = cfg['gpu'] if 'gpu' in cfg else False
        self.suboptimizer = VariateTrpoUpdater(nets, cfg)
        self.optimizer_way = self._fit_q if cfg['opt'] == 'fitq' else self._min_var
        self.call = 0

    def _get_min_var_grad(self):
        log_probs = self.policy.get_log_prob(self.states, self.actions)
        prob_ratio = torch.exp(log_probs - self.fixed_log_probs)
        advantage_term = (self.variate(self.states, self.actions) - self.advantages) * prob_ratio
        action_mean, action_log_std = self.policy(self.states)
        variate_term = self.variate(self.states, action_mean + action_log_std.exp() * self.xi) * prob_ratio.data
        action_loss = (advantage_term - variate_term).mean()

        flat_grad = []
        grads = torch.autograd.grad(action_loss, self.policy.parameters())
        for grad in grads:
            grad.requires_grad = True
            flat_grad.append(grad.view(-1))
        flat_grad = torch.cat(flat_grad)
        return flat_grad.pow(2).mean()

    def _min_var(self, batch, log):
        """
        Updating variate and value networks by minimizing least square of q, i.e.

            min_w sum_t [ delta_policy log(policy(a_t | s_t)) * (Q(s_t, a_t) - phi_w(s_t, a_t)) +
                          delta_policy policy(s_t, xi_t) delta_a phi_w(s_t, a_t)
                         ] ** 2
        """
        variate_loss = self._get_min_var_grad()
        log["variate_loss/min_var"] = variate_loss.item()

        self.optimizer_variate.zero_grad()
        variate_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.variate.parameters(), 40)
        self.optimizer_variate.step()
        return log

    def _fit_q(self, batch, log):
        """
        Updating variate and value networks by minimizing L2 loss, i.e.

            min_w sum_t (Phi_w(s_t, a_t) - R_t) ** 2

        where Phi_w = value + phi_w.
        """
        rewards = batch["rewards"]
        variate_loss = (self.value(self.states) + self.variate(self.states, self.actions) - rewards).pow(2).mean()
        log["variate_loss/fit_q"] = variate_loss.item()

        self.optimizer_variate.zero_grad()
        variate_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.variate.parameters(), 40)
        self.optimizer_variate.step()
        return log

    def __call__(self, batch, log, *args, **kwargs):
        self.call += 1
        self.states = batch["states"]
        self.actions = batch["actions"]
        self.advantages = batch["advantages"]
        with torch.no_grad():
            self.mu, self.log_std = self.policy(self.states)
            self.fixed_log_probs = self.policy.get_log_prob(self.states, self.actions).detach()
            self.xi = (self.actions - self.mu) / self.log_std.exp()

        log = self.suboptimizer(batch, log, *args, **kwargs)
        if self.call % self.optim_variate_iterval == 1:
            for _ in range(self.optim_variate_iternum):
                log = self.optimizer_way(batch, log)
        return log

    def state_dict(self):
        return {'optimizer_policy': self.optimizer_policy.state_dict(),
                'optimizer_value': self.optimizer_value.state_dict(),
                'optimizer_variate': self.optimizer_variate.state_dict(),
                'call': self.call}

    def load_state_dict(self, state_dict):
        self.optimizer_policy.load_state_dict(state_dict['optimizer_policy'])
        self.optimizer_value.load_state_dict(state_dict['optimizer_value'])
        self.optimizer_variate.load_state_dict(state_dict['optimizer_variate'])
        self.call = state_dict['call'] + 1
