import torch


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
        values_pred = self.value(states)
        value_loss = (values_pred - value_targets).pow(2).mean()
        log["value_loss"] = value_loss.item()

        # weight decay
        if self.l2_reg > 0:
            for param in self.value.parameters():
                value_loss += param.pow(2).sum() * self.l2_reg
            log["value_loss_l2r"] = value_loss.item()

        self.optimizer_value.zero_grad()
        value_loss.backward()
        self.optimizer_value.step()

        # update policy
        policy_loss = -(self.policy.get_log_prob(states, actions) * advantages).mean()
        log["policy_loss"] = policy_loss.item()

        self.optimizer_policy.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 40)
        self.optimizer_policy.step()

        return log

    def state_dict(self):
        return {'optimizer_policy': self.optimizer_policy.state_dict(),
                'optimizer_value': self.optimizer_value.state_dict()}

    def load_state_dict(self, state_dict):
        self.optimizer_policy.load_state_dict(state_dict['optimizer_policy'])
        self.optimizer_value.load_state_dict(state_dict['optimizer_value'])
