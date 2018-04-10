import torch
from torch.autograd import Variable


def a2c_step(policy_net, value_net, optimizer_policy, optimizer_value, states, actions, returns, advantages, l2_reg):

    """update critic"""
    values_target = Variable(returns)
    values_pred = value_net(Variable(states))
    value_loss = (values_pred - values_target).pow(2).mean()
    # weight decay
    for param in value_net.parameters():
        value_loss += param.pow(2).sum() * l2_reg
    optimizer_value.zero_grad()
    value_loss.backward()
    optimizer_value.step()

    """update policy"""
    log_probs = policy_net.get_log_prob(Variable(states), Variable(actions))
    policy_loss = -(log_probs * Variable(advantages)).mean()
    optimizer_policy.zero_grad()
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm(policy_net.parameters(), 40)
    optimizer_policy.step()


class A2cUpdater(object):
    def __init__(self, policy_net, value_net, optimizer_policy, optimizer_value, cfg):
        self.policy = policy_net
        self.value = value_net
        self.optimizer_policy = optimizer_policy
        self.optimizer_value = optimizer_value
        self.l2_reg = cfg["l2_reg"]

    def __call__(self, batch, log, *args, **kwargs):
        states = batch["states"]
        actions = batch["actions"]
        value_targets = batch["value_targets"]
        advantages = batch["advantages"]
        # Careful choice of lrs is necessary for good performance

        # update critic
        values_targets = Variable(value_targets)
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

        # update policy
        log_probs = self.policy.get_log_prob(Variable(states), Variable(actions))
        policy_loss = -(log_probs * Variable(advantages)).mean()
        log["policy loss"] = policy_loss.data[0]

        self.optimizer_policy.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm(self.policy.parameters(), 40)
        self.optimizer_policy.step()

        return log
