from utils.torchs import *
from utils.distribution import *
import scipy.optimize
import math


class FTrpoUpdater(object):
    def __init__(self, policy_net, value_net, cfg, use_fim=False):
        self.policy_net = policy_net
        self.value_net = value_net
        self.gpu = cfg["gpu"]
        self.max_kl = cfg["max_kl"]
        self.damping = cfg["damping"]
        self.l2_reg = cfg["l2_reg"]
        self.nsteps = cfg["nsteps"] if "nsteps" in cfg else 10
        self.alpha = cfg["alpha"]
        self.use_fim = use_fim
        self.Fvp = self.fvp_direct

    def get_value_loss(self, flat_params):
        set_flat_params_to(self.value_net, np_to_tensor(flat_params))
        for param in self.value_net.parameters():
            if param.grad is not None:
                param.grad.data.fill_(0)
        values_pred = self.value_net(self.states)
        value_loss = (values_pred - self.values_targets).pow(2).mean()

        # weight decay
        for param in self.value_net.parameters():
            value_loss += param.pow(2).sum() * self.l2_reg
        value_loss.backward()
        return value_loss.item(), get_flat_grad_from(self.value_net.parameters()).detach().cpu().numpy()

    def get_policy_loss(self):
        log_probs = self.policy_net.get_log_prob(self.states, self.actions)
        action_loss = - self.advantages * torch.exp(log_probs - self.fixed_log_probs)
        return action_loss.mean()

    def conjugate_gradients(self, b, rdotr_tol=1e-10):
        x = zeros(b.size(), gpu=use_gpu and self.gpu)
        r = b.clone()
        p = b.clone()
        rdotr = torch.dot(r, r)
        for i in range(self.nsteps):
            avp = self.Fvp(p)
            alpha = rdotr / torch.dot(p, avp)
            x += alpha * p
            r -= alpha * avp
            new_rdotr = torch.dot(r, r)
            betta = new_rdotr / rdotr
            p = r + betta * p
            rdotr = new_rdotr
            if rdotr < rdotr_tol:
                break
        return x

    def line_search(self, x, fullstep, expected_improve_full, max_backtracks=10, accept_ratio=0.1):
        with torch.no_grad():
            fval = self.get_policy_loss().item()

            for stepfrac in [.5 ** x for x in range(max_backtracks)]:
                x_new = x + stepfrac * fullstep
                set_flat_params_to(self.policy_net, x_new)
                fval_new = self.get_policy_loss().item()
                actual_improve = fval - fval_new
                expected_improve = expected_improve_full * stepfrac
                ratio = actual_improve / expected_improve

                if ratio > accept_ratio:
                    return True, x_new
            return False, x

    # directly compute Hessian*vector from f-divergence
    def fvp_direct(self, v):
        if self.alpha == 1:
            fd = self.policy_net.get_kl(self.states).mean()
        else:
            action_mean, action_log_std = self.policy_net(self.states)
            samples_actions = torch.normal(mean=action_mean, std=action_log_std).detach()
            samples_density = normal_log_density(samples_actions, action_mean, action_log_std).exp()
            sample_ratio = samples_density.detach()/samples_density
            if self.alpha == 2:
                fd = 0.5 * (sample_ratio-1).pow(2).mean()
            elif self.alpha == -1:
                fd = 0.5 * ((sample_ratio-1).pow(2)/sample_ratio).mean()
            elif self.alpha == 0.5:
                fd = 2 * (torch.sqrt(sample_ratio)-1).pow(2).mean()
            else:
                fd = ((sample_ratio**self.alpha-1-self.alpha*(sample_ratio-1))/self.alpha/(self.alpha-1)).mean()

        grads = torch.autograd.grad(fd, self.policy_net.parameters(), create_graph=True)
        flat_grad_fd = torch.cat([grad.view(-1) for grad in grads])

        fd_v = (flat_grad_fd * v).sum()
        grads = torch.autograd.grad(fd_v, self.policy_net.parameters())
        flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).data

        return flat_grad_grad_kl + v * self.damping

    def __call__(self, batch, log, *args, **kwargs):
        self.states = batch["states"]
        self.actions = batch["actions"]
        self.advantages = batch["advantages"]
        with torch.no_grad():
            self.fixed_log_probs = self.policy_net.get_log_prob(self.states, self.actions).data

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

    def state_dict(self):
        return {"max_kl": self.max_kl,
                "damping": self.damping,
                "l2_reg": self.l2_reg,
                "nsteps": self.nsteps,
                "use_fim": self.use_fim}

    def load_state_dict(self, state_dict):
        self.max_kl = state_dict["max_kl"]
        self.damping = state_dict["damping"]
        self.l2_reg = state_dict["l2_reg"]
        self.nsteps = state_dict["nsteps"]
        self.use_fim = state_dict["use_fim"]
        self.Fvp = self.fvp_fim if self.use_fim else self.fvp_direct
