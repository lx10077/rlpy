from utils.torchs import *
from utils.tools import estimate_advantages


class SacUpdater(object):
    def __init__(self, policy_net, value_net, optimizer_policy, optimizer_value, cfg):
        self.policy = policy_net
        self.value = value_net
        self.optimizer_policy = optimizer_policy
        self.optimizer_value = optimizer_value
        self.cfg = cfg

        assert "lambd" in cfg
        self.lambd = cfg["lambd"]
        self.l2_reg = cfg["l2_reg"]
        self.lr = cfg["learning_rate"]
        self.clip_epsilon = cfg["clip_epsilon"]
        self.max_iter_num = cfg["max_iter_num"]
        self.optim_epochs = cfg["optim_epochs"]
        self.optim_batch_size = cfg["optim_batch_size"]
        self.optim_value_iternum = cfg["optim_value_iternum"]

    def update_value(self, states, values_targets, log):
        for _ in range(self.optim_value_iternum):
            values_pred = self.value(states)
            value_loss = (values_pred - values_targets).pow(2).mean()
            log["value_loss"] = value_loss.item()

            # weight decay
            if self.l2_reg > 0:
                for param in self.value.parameters():
                    value_loss += param.pow(2).sum() * self.l2_reg
                log["value_loss_w_l2r"] = value_loss.item()

            self.optimizer_value.zero_grad()
            value_loss.backward()
            self.optimizer_value.step()
        return log

    def update_policy(self, states, actions, advantages, fixed_log_probs, log):
        log_probs = self.policy.get_log_prob(states, actions)
        ratio = torch.exp(log_probs - fixed_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.curr_clip_epsilon, 1.0 + self.curr_clip_epsilon) * advantages
        entropy = self.policy.get_entropy(states).mean()
        log["entropy"] = entropy.item()
        policy_surr = - torch.min(surr1, surr2).mean() - self.lambd * entropy
        # policy_surr = -torch.cat([surr1, surr2], 1).min(1)[0].mean()
        log["policy_surr"] = policy_surr.item()

        self.optimizer_policy.zero_grad()
        policy_surr.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 40)
        self.optimizer_policy.step()
        return log

    def __call__(self, batch, log, iter_i, *args, **kwargs):
        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        masks = batch["masks"]
        values = self.value(states)
        fixed_log_probs = self.policy.get_log_prob(states, actions).detach()
        composite_rewards = rewards - self.lambd * fixed_log_probs.view(-1)
        advantages, value_targets = estimate_advantages(composite_rewards, masks, values, self.cfg["gamma"],
                                                        self.cfg["tau"], use_gpu & self.cfg["gpu"])

        lr_mult = max(1.0 - float(iter_i) / self.max_iter_num, 0)
        self.optimizer_policy.lr = self.lr * lr_mult
        self.optimizer_value.lr = self.lr * lr_mult
        self.curr_clip_epsilon = self.clip_epsilon * lr_mult
        self.lambd = self.lambd * lr_mult
        log["clip_eps"] = self.curr_clip_epsilon

        for _ in range(self.optim_epochs):
            log = self.update_value(values, value_targets, log)
            log = self.update_policy(states, actions, advantages, fixed_log_probs, log)
        return log

    def state_dict(self):
        return {'optimizer_policy': self.optimizer_policy.state_dict(),
                'optimizer_value': self.optimizer_value.state_dict()}

    def load_state_dict(self, state_dict):
        self.optimizer_policy.load_state_dict(state_dict['optimizer_policy'])
        self.optimizer_value.load_state_dict(state_dict['optimizer_value'])
