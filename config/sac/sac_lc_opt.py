from utils.torchs import *


def estimate_advantages(rewards, masks, values, gamma, tau, use_gpu):
    if use_gpu:
        rewards, masks, values = rewards.cpu(), masks.cpu(), values.cpu()
    tensor_type = type(rewards)
    deltas = tensor_type(rewards.size(0), 1)
    advantages = tensor_type(rewards.size(0), 1)

    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        deltas[i] = rewards[i] + gamma * prev_value * masks[i] - values[i]
        advantages[i] = deltas[i] + gamma * tau * prev_advantage * masks[i]

        prev_value = values[i, 0]
        prev_advantage = advantages[i, 0]

    returns = values + advantages
    advantages = (advantages - advantages.mean()) / advantages.std()

    if use_gpu:
        advantages, returns = advantages.cuda(), returns.cuda()
    return advantages, returns


class SacLcUpdater(object):
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
        with torch.no_grad():
            values = self.value(states)
            fixed_log_probs = self.policy.get_log_prob(states, actions).data
            composite_rewards = rewards - self.lambd * fixed_log_probs.view(-1)
            advantages, value_targets = estimate_advantages(composite_rewards, masks, values, self.cfg["gamma"],
                                                            self.cfg["tau"], use_gpu & self.cfg["gpu"])

        num_sample = states.shape[0]
        optim_iter_num = int(np.ceil(num_sample / self.optim_batch_size))
        lr_mult = max(1.0 - float(iter_i) / self.max_iter_num, 0)
        self.optimizer_policy.lr = self.lr * lr_mult
        self.optimizer_value.lr = self.lr * lr_mult
        self.curr_clip_epsilon = self.clip_epsilon * lr_mult
        self.lambd = self.lambd * lr_mult
        log["clip_eps"] = self.curr_clip_epsilon

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

    def state_dict(self):
        return {'optimizer_policy': self.optimizer_policy.state_dict(),
                'optimizer_value': self.optimizer_value.state_dict()}

    def load_state_dict(self, state_dict):
        self.optimizer_policy.load_state_dict(state_dict['optimizer_policy'])
        self.optimizer_value.load_state_dict(state_dict['optimizer_value'])
