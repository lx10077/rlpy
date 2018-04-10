from utils.torch import *


def ppo_step(policy_net, value_net, optimizer_policy, optimizer_value, optim_value_iternum, states, actions,
             returns, advantages, fixed_log_probs, lr_mult, lr, clip_epsilon, l2_reg):

    optimizer_policy.lr = lr * lr_mult
    optimizer_value.lr = lr * lr_mult
    clip_epsilon = clip_epsilon * lr_mult

    """update critic"""
    values_target = Variable(returns)
    for _ in range(optim_value_iternum):
        values_pred = value_net(Variable(states))
        value_loss = (values_pred - values_target).pow(2).mean()
        # weight decay
        for param in value_net.parameters():
            value_loss += param.pow(2).sum() * l2_reg
        optimizer_value.zero_grad()
        value_loss.backward()
        optimizer_value.step()

    """update policy"""
    advantages_var = Variable(advantages)
    log_probs = policy_net.get_log_prob(Variable(states), Variable(actions))
    ratio = torch.exp(log_probs - Variable(fixed_log_probs))
    surr1 = ratio * advantages_var
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages_var
    policy_surr = -torch.min(surr1, surr2).mean()
    optimizer_policy.zero_grad()
    policy_surr.backward()
    torch.nn.utils.clip_grad_norm(policy_net.parameters(), 40)
    optimizer_policy.step()


class ClipPpoUpdater(object):
    def __init__(self, policy_net, value_net, optimizer_policy, optimizer_value, cfg):
        self.policy = policy_net
        self.value = value_net
        self.optimizer_policy = optimizer_policy
        self.optimizer_value = optimizer_value
        self.cfg = cfg

        self.l2_reg = cfg["l2_reg"]
        self.lr = cfg["learning_rate"]
        self.clip_epsilon = cfg["clip_epsilon"]
        self.max_iter_num = cfg["max_iter_num"]
        self.optim_epochs = cfg["optim_epochs"]
        self.optim_batch_size = cfg["optim_batch_size"]
        self.optim_value_iternum = cfg["optim_value_iternum"]

    def update_value(self, states, values_targets, log):
        values_targets = Variable(values_targets)
        for _ in range(self.optim_value_iternum):
            values_pred = self.value(Variable(states))
            value_loss = (values_pred - values_targets).pow(2).mean()
            log["value loss"] = value_loss.data[0]

            # weight decay
            if self.l2_reg > 0:
                for param in self.value.parameters():
                    value_loss += param.pow(2).sum() * self.l2_reg
                log["value loss w l2r"] = value_loss.data[0]

            self.optimizer_value.zero_grad()
            value_loss.backward()
            self.optimizer_value.step()
        return log

    def update_policy(self, states, actions, advantages, fixed_log_probs, log):
        advantages_var = Variable(advantages)
        log_probs = self.policy.get_log_prob(Variable(states), Variable(actions))
        ratio = torch.exp(log_probs - Variable(fixed_log_probs))
        surr1 = ratio * advantages_var
        surr2 = torch.clamp(ratio, 1.0 - self.curr_clip_epsilon, 1.0 + self.curr_clip_epsilon) * advantages_var
        policy_surr = -torch.min(surr1, surr2).mean()
        # policy_surr = -torch.cat([surr1, surr2], 1).min(1)[0].mean()
        log["policy_surr"] = policy_surr.data[0]

        self.optimizer_policy.zero_grad()
        policy_surr.backward()
        torch.nn.utils.clip_grad_norm(self.policy.parameters(), 40)
        self.optimizer_policy.step()
        return log

    def __call__(self, batch, log, iter_i, *args, **kwargs):
        states = batch["states"]
        actions = batch["actions"]
        advantages = batch["advantages"]
        value_targets = batch["value_targets"]
        fixed_log_probs = self.policy.get_log_prob(Variable(states, volatile=True), Variable(actions)).data

        num_sample = states.shape[0]
        optim_iter_num = int(np.ceil(num_sample / self.optim_batch_size))

        lr_mult = max(1.0 - float(iter_i) / self.max_iter_num, 0)
        self.optimizer_policy.lr = self.lr * lr_mult
        self.optimizer_value.lr = self.lr * lr_mult
        self.curr_clip_epsilon = self.clip_epsilon * lr_mult
        log["clip eps"] = self.curr_clip_epsilon
        log["lr"] = self.lr * lr_mult

        for _ in range(self.optim_epochs):
            perm = np.arange(num_sample)
            np.random.shuffle(perm)
            perm = np_to_tensor(perm).long()
            if use_gpu and self.cfg["gpu"]:
                perm = perm.cuda()

            # it is important to shuffle samples
            states, actions, value_targets, advantages, fixed_log_probs = \
                states[perm], actions[perm], value_targets[perm], advantages[perm], fixed_log_probs[perm]

            for i in range(optim_iter_num):
                # do minibatch optimization
                ind = slice(i * self.optim_batch_size, min((i + 1) * self.optim_batch_size, num_sample))
                states_b, actions_b, advantages_b, value_targets_b, fixed_log_probs_b = \
                    states[ind], actions[ind], advantages[ind], value_targets[ind], fixed_log_probs[ind]

                log = self.update_value(states_b, value_targets_b, log)
                log = self.update_policy(states_b, actions_b, advantages_b, fixed_log_probs_b, log)

        return log
