from utils.torch import *
import scipy.optimize
import math


def conjugate_gradients(Avp_f, b, nsteps, rdotr_tol=1e-10):
    x = zeros(b.size())
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for i in range(nsteps):
        Avp = Avp_f(p)
        alpha = rdotr / torch.dot(p, Avp)
        x += alpha * p
        r -= alpha * Avp
        new_rdotr = torch.dot(r, r)
        betta = new_rdotr / rdotr
        p = r + betta * p
        rdotr = new_rdotr
        if rdotr < rdotr_tol:
            break
    return x


def line_search(model, f, x, fullstep, expected_improve_full, max_backtracks=10, accept_ratio=0.1):
    fval = f(True).data[0]

    for stepfrac in [.5**x for x in range(max_backtracks)]:
        x_new = x + stepfrac * fullstep
        set_flat_params_to(model, x_new)
        fval_new = f(True).data[0]
        actual_improve = fval - fval_new
        expected_improve = expected_improve_full * stepfrac
        ratio = actual_improve / expected_improve

        if ratio > accept_ratio:
            return True, x_new
    return False, x


def trpo_step(policy_net, value_net, states, actions, returns, advantages, max_kl, damping, l2_reg, use_fim=False):

    """update critic"""
    values_target = Variable(returns)

    def get_value_loss(flat_params):
        set_flat_params_to(value_net, torch.Tensor(flat_params))
        for param in value_net.parameters():
            if param.grad is not None:
                param.grad.data.fill_(0)
        values_pred = value_net(Variable(states))
        value_loss = (values_pred - values_target).pow(2).mean()

        # weight decay
        for param in value_net.parameters():
            value_loss += param.pow(2).sum() * l2_reg
        value_loss.backward()
        return value_loss.data.cpu().numpy()[0], get_flat_grad_from(value_net.parameters()).data.cpu().numpy()

    flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(get_value_loss,
                                                            get_flat_params_from(value_net).cpu().numpy(),
                                                            maxiter=25)
    set_flat_params_to(value_net, torch.Tensor(flat_params))

    """update policy"""
    fixed_log_probs = policy_net.get_log_prob(Variable(states, volatile=True), Variable(actions)).data
    """define the loss function for TRPO"""
    def get_loss(volatile=False):
        log_probs = policy_net.get_log_prob(Variable(states, volatile=volatile), Variable(actions))
        action_loss = -Variable(advantages) * torch.exp(log_probs - Variable(fixed_log_probs))
        return action_loss.mean()

    """use fisher information matrix for Hessian*vector"""
    def Fvp_fim(v):
        M, mu, info = policy_net.get_fim(Variable(states))
        mu = mu.view(-1)
        filter_input_ids = set() if policy_net.is_disc_action else set([info['std_id']])

        t = Variable(ones(mu.size()), requires_grad=True)
        mu_t = (mu * t).sum()
        Jt = compute_flat_grad(mu_t, policy_net.parameters(), filter_input_ids=filter_input_ids, create_graph=True)
        Jtv = (Jt * Variable(v)).sum()
        Jv = torch.autograd.grad(Jtv, t, retain_graph=True)[0]
        MJv = Variable(M * Jv.data)
        mu_MJv = (MJv * mu).sum()
        JTMJv = compute_flat_grad(mu_MJv, policy_net.parameters(), filter_input_ids=filter_input_ids, retain_graph=True).data
        JTMJv /= states.shape[0]
        if not policy_net.is_disc_action:
            std_index = info['std_index']
            JTMJv[std_index: std_index + M.shape[0]] += 2 * v[std_index: std_index + M.shape[0]]
        return JTMJv + v * damping

    """directly compute Hessian*vector from KL"""
    def Fvp_direct(v):
        kl = policy_net.get_kl(Variable(states))
        kl = kl.mean()

        grads = torch.autograd.grad(kl, policy_net.parameters(), create_graph=True)
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

        kl_v = (flat_grad_kl * Variable(v)).sum()
        grads = torch.autograd.grad(kl_v, policy_net.parameters())
        flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).data

        return flat_grad_grad_kl + v * damping

    Fvp = Fvp_fim if use_fim else Fvp_direct

    loss = get_loss()
    grads = torch.autograd.grad(loss, policy_net.parameters())
    loss_grad = torch.cat([grad.view(-1) for grad in grads]).data
    stepdir = conjugate_gradients(Fvp, -loss_grad, 10)

    shs = 0.5 * (stepdir.dot(Fvp(stepdir)))
    lm = math.sqrt(max_kl / shs)
    fullstep = stepdir * lm
    expected_improve = -loss_grad.dot(fullstep)

    prev_params = get_flat_params_from(policy_net)
    success, new_params = line_search(policy_net, get_loss, prev_params, fullstep, expected_improve)
    set_flat_params_to(policy_net, new_params)

    return success


class TrpoUpdater(object):
    def __init__(self, policy_net, value_net, max_kl, damping, l2_reg, nsteps, use_fim=False):
        self.policy_net = policy_net
        self.value_net = value_net
        self.max_kl = max_kl
        self.damping = damping
        self.l2_reg = l2_reg
        self.use_fim = use_fim
        self.nsteps = nsteps
        self.Fvp = self.Fvp_fim if use_fim else self.Fvp_direct

    def get_value_loss(self, flat_params):
        set_flat_params_to(self.value_net, np_to_tensor(flat_params))
        for param in self.value_net.parameters():
            if param.grad is not None:
                param.grad.data.fill_(0)
        values_pred = self.value_net(Variable(self.states))
        value_loss = (values_pred - self.values_targets).pow(2).mean()

        # weight decay
        for param in self.value_net.parameters():
            value_loss += param.pow(2).sum() * self.l2_reg
        value_loss.backward()
        return value_loss.data.cpu().numpy()[0], get_flat_grad_from(self.value_net.parameters()).data.cpu().numpy()

    def get_policy_loss(self, volatile=False):
        log_probs = self.policy_net.get_log_prob(Variable(self.states, volatile=volatile), Variable(self.actions))
        action_loss = -Variable(self.advantages) * torch.exp(log_probs - Variable(self.fixed_log_probs))
        return action_loss.mean()

    def conjugate_gradients(self, Avp_f, b, rdotr_tol=1e-10):
        x = torch.zeros(b.size())
        r = b.clone()
        p = b.clone()
        rdotr = torch.dot(r, r)
        for i in range(self.nsteps):
            Avp = Avp_f(p)
            alpha = rdotr / torch.dot(p, Avp)
            x += alpha * p
            r -= alpha * Avp
            new_rdotr = torch.dot(r, r)
            betta = new_rdotr / rdotr
            p = r + betta * p
            rdotr = new_rdotr
            if rdotr < rdotr_tol:
                break
        return x

    def line_search(self, x, fullstep, expected_improve_full, max_backtracks=10, accept_ratio=0.1):
        fval = self.get_policy_loss(True).data[0]

        for stepfrac in [.5 ** x for x in range(max_backtracks)]:
            x_new = x + stepfrac * fullstep
            set_flat_params_to(self.policy_net, x_new)
            fval_new = self.get_policy_loss(True).data[0]
            actual_improve = fval - fval_new
            expected_improve = expected_improve_full * stepfrac
            ratio = actual_improve / expected_improve

            if ratio > accept_ratio:
                return True, x_new
        return False, x

    # use fisher information matrix for Hessian * vector
    def Fvp_fim(self, v):
        M, mu, info = self.policy_net.get_fim(Variable(self.states))
        mu = mu.view(-1)
        filter_input_ids = set() if self.policy_net.is_disc_action else set(info['std_id'])

        t = Variable(torch.ones(mu.size()), requires_grad=True)
        mu_t = (mu * t).sum()
        Jt = compute_flat_grad(mu_t, self.policy_net.parameters(),
                               filter_input_ids=filter_input_ids, create_graph=True)
        Jtv = (Jt * Variable(v)).sum()
        Jv = torch.autograd.grad(Jtv, t, retain_graph=True)[0]
        MJv = Variable(M * Jv.data)
        mu_MJv = (MJv * mu).sum()
        JTMJv = compute_flat_grad(mu_MJv, self.policy_net.parameters(),
                                  filter_input_ids=filter_input_ids, retain_graph=True).data
        JTMJv /= self.states.shape[0]
        if not self.policy_net.is_disc_action:
            std_index = info['std_index']
            JTMJv[std_index: std_index + M.shape[0]] += 2 * v[std_index: std_index + M.shape[0]]
        return JTMJv + v * self.damping

    # directly compute Hessian*vector from KL
    def Fvp_direct(self, v):
        kl = self.policy_net.get_kl(Variable(self.states))
        kl = kl.mean()

        grads = torch.autograd.grad(kl, self.policy_net.parameters(), create_graph=True)
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

        kl_v = (flat_grad_kl * Variable(v)).sum()
        grads = torch.autograd.grad(kl_v, self.policy_net.parameters())
        flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).data

        return flat_grad_grad_kl + v * self.damping

    def __call__(self, batchs, log):
        self.states = batchs["states"]
        self.actions = batchs["actions"]
        self.advantages = batchs["advantages"]
        self.fixed_log_probs = self.policy_net.get_log_prob(Variable(self.states), Variable(self.actions)).data

        # update the value networks by L-BFGS
        self.values_targets = Variable(batchs["value_targets"])
        flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(self.get_value_loss,
                                                                get_flat_params_from(self.value_net).cpu().numpy(),
                                                                maxiter=25)
        set_flat_params_to(self.value_net, np_to_tensor(flat_params))
        value_loss = (self.value_net(Variable(self.states)) - self.values_targets).pow(2).mean()
        log["loss value"] = value_loss.data[0]

        # update the policy networks by trust region gradient
        policy_loss = self.get_policy_loss()
        grads = torch.autograd.grad(policy_loss, self.policy_net.parameters())
        loss_grad = torch.cat([grad.view(-1) for grad in grads]).data
        stepdir = conjugate_gradients(self.Fvp, -loss_grad, 10)

        shs = 0.5 * (stepdir.dot(self.Fvp(stepdir)))
        lm = math.sqrt(self.max_kl / shs)
        fullstep = stepdir * lm
        expected_improve = -loss_grad.dot(fullstep)
        prev_params = get_flat_params_from(self.policy_net)
        success, new_params = self.line_search(prev_params, fullstep, expected_improve)
        set_flat_params_to(self.policy_net, new_params)
