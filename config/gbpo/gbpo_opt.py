from utils.torchs import *
from torch.optim import Adam


class GbpoUpdater(object):
    def __init__(self, policy, temporary_policy, value, optimizer_policy, optimizer_value, cfg):
        self.policy = policy
        self.temporary_policy = temporary_policy
        self.value = value
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
        log_probs = self.temporary_policy.get_log_prob(states, actions)
        ratio = torch.exp(log_probs - fixed_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.curr_clip_epsilon, 1.0 + self.curr_clip_epsilon) * advantages
        policy_surr = -torch.min(surr1, surr2).mean()
        # policy_surr = -torch.cat([surr1, surr2], 1).min(1)[0].mean()
        log["policy_surr"] = policy_surr.item()

        self.optimizer_policy.zero_grad()
        policy_surr.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 40)
        self.optimizer_policy.step()
        return log

    def mimic_policy(self, new_policy, states, log):
        optimizer = Adam(new_policy.parameters(), lr=self.lr)
        teacher_mean, teacher_log_std = self.temporary_policy(states)

        for _ in range(self.optim_epochs * 5):
            student_mean, student_log_std = new_policy(states)
            mimic_loss = (student_mean - teacher_mean).pow(2).mean() + (
                    student_log_std - teacher_log_std).pow(2).mean()

            optimizer.zero_grad()
            mimic_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(new_policy.parameters(), 40)
            optimizer.step()

            log["mimic_policy_loss"] = mimic_loss.item()
            if mimic_loss.item() < 1e-5:
                break
        return new_policy, log

    def __call__(self, batch, log, iter_i, *args, **kwargs):
        states = batch["states"]
        actions = batch["actions"]
        advantages = batch["advantages"]
        value_targets = batch["value_targets"]
        with torch.no_grad():
            fixed_log_probs = self.temporary_policy.get_log_prob(states, actions).data

        num_sample = states.shape[0]
        optim_iter_num = int(np.ceil(num_sample / self.optim_batch_size))

        lr_mult = max(1.0 - float(iter_i) / self.max_iter_num, 0)
        self.optimizer_policy.lr = self.lr * lr_mult
        self.optimizer_value.lr = self.lr * lr_mult
        self.curr_clip_epsilon = self.clip_epsilon * lr_mult
        log["clip_eps"] = self.curr_clip_epsilon
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

        new_param = get_flat_params_from(self.policy)
        self.policy.add_policy(new_param)

        if self.policy.is_full:
            print('full')
            new_policy = self.policy.generate_policy()
            learned_policy, log = self.mimic_policy(new_policy, states, log)
            learned_policy_param = get_flat_params_from(learned_policy)
            self.policy.clear_policy(learned_policy_param)
        return log

    def state_dict(self):
        return {'optimizer_policy': self.optimizer_policy.state_dict(),
                'optimizer_value': self.optimizer_value.state_dict()}

    def load_state_dict(self, state_dict):
        self.optimizer_policy.load_state_dict(state_dict['optimizer_policy'])
        self.optimizer_value.load_state_dict(state_dict['optimizer_value'])
