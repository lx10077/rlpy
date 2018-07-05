from core.trpo import TrpoUpdater
from utils.torchs import *


class VariateTrpoUpdater(TrpoUpdater):
    def __init__(self, nets, cfg, use_fim=False):
        self.variate = nets['adcv']
        super(VariateTrpoUpdater, self).__init__(nets['policy'], nets['value'], cfg, use_fim=use_fim)

    def get_policy_loss(self):
        log_probs = self.policy_net.get_log_prob(self.states, self.actions)
        term = self.variate(self.states, self.policy_net(self.states))
        action_loss = self.variate(self.states, self.actions) - self.advantages - term
        action_loss *= torch.exp(log_probs - self.fixed_log_probs)
        return action_loss.mean()


class VariateUpdater(object):
    def __init__(self, nets, optimizers, cfg):
        self.policy = nets['policy']
        self.value = nets['value']
        self.variate = nets['adcv']
        self.optimizer_policy = optimizers['optimizer_policy']
        self.optimizer_value = optimizers['optimizer_value']
        self.optimizer_variate = optimizers['optimizer_variate']
        self.optim_variate_iternum = cfg['optim_variate_iternum'] if 'optim_variate_iternum' in cfg else 1
        self.gpu = cfg['gpu'] if 'gpu' in cfg else False
        self.suboptimizer = VariateTrpoUpdater(nets, cfg)

    def _get_min_var_grad(self):
        log_probs = self.policy.get_log_prob(self.states, self.actions)
        term = self.variate(self.states, self.policy(self.states))
        action_loss = (self.variate(self.states, self.actions) - self.advantages) * log_probs - term
        action_loss.backward()
        grad = get_flat_grad_from(action_loss, self.policy)
        return grad.pow(2).mean()

    def _min_var(self, batch, log):
        """
        Updating adcv and value networks by minimizing least square of q, i.e.

        min_w sum_t [ delta_policy log(policy(a_t | s_t)) * (Q(s_t, a_t) - phi_w(s_t, a_t)) +
                      delta_policy policy(s_t, xi_t) delta_a phi_w(s_t, a_t)
                     ] ** 2
        """
        variate_loss = self._get_min_var_grad()
        log["variate_loss/min_var"] = variate_loss.data[0]

        self.optimizer_variate.zero_grad()
        variate_loss.backward()
        torch.nn.utils.clip_grad_norm(self.variate.parameters(), 40)
        self.optimizer_variate.step()
        return log

    def _fit_q(self, batch, log):
        """
        Updating adcv and value networks by minimizing least square of q, i.e.

            min_w sum_t (Phi_w(s_t, a_t) - R_t) ** 2

        where Phi_w = value + phi_w.
        """
        rewards = batch["rewards"]
        variate_loss = (self.variate(self.states, self.actions) - rewards).mean()
        log["variate_loss/fit_q"] = variate_loss.data[0]

        self.optimizer_variate.zero_grad()
        variate_loss.backward()
        torch.nn.utils.clip_grad_norm(self.variate.parameters(), 40)
        self.optimizer_variate.step()
        return log

    def __call__(self, batch, log, *args, **kwargs):
        self.states = batch["states"]
        self.actions = batch["actions"]
        self.advantages = batch["advantages"]
        with torch.no_grad():
            self.fixed_log_probs = self.policy.get_log_prob(self.states, self.actions).detach()

        log = self.suboptimizer(batch, log, *args, **kwargs)
        for _ in range(self.optim_variate_iternum):
            log = self._fit_q(batch, log)
        return log

    def state_dict(self):
        return {'optimizer_policy': self.optimizer_policy.state_dict(),
                'optimizer_value': self.optimizer_variate.state_dict()}

    def load_state_dict(self, state_dict):
        self.optimizer_policy.load_state_dict(state_dict['optimizer_policy'])
        self.optimizer_value.load_state_dict(state_dict['optimizer_value'])
