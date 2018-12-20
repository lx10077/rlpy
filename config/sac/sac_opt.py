from utils.torchs import *


class SacUpdater(object):
    def __init__(self, policy_net, value_net, qvalue_net,
                 optimizer_policy, optimizer_value, optimizer_qvalue,
                 cfg):
        self.policy = policy_net
        self.value = value_net
        self.qvalue_net = qvalue_net
        self.optimizer_policy = optimizer_policy
        self.optimizer_value = optimizer_value
        self.optimizer_qvalue = optimizer_qvalue
        self.cfg = cfg

        self.l2_reg = cfg["l2_reg"]
        self.lr = cfg["learning_rate"]
        self.max_iter_num = cfg["max_iter_num"]
        self.optim_epochs = cfg["optim_epochs"]
        self.optim_batch_size = cfg["optim_batch_size"]
        self.optim_value_iternum = cfg["optim_value_iternum"]
        self.optim_qvalue_iternum = cfg["optim_qvalue_iternum"]

    def update_qvalue(self, states, actions, values_targets, log):
        for _ in range(self.optim_qvalue_iternum):
            qvalues_pred = self.qvalue_net(states, actions)
            qvalue_loss = (qvalues_pred - values_targets).pow(2).mean()
            log["qvalue_loss"] = qvalue_loss.item()

            # weight decay
            if self.l2_reg > 0:
                for param in self.value.parameters():
                    qvalue_loss += param.pow(2).sum() * self.l2_reg
                log["qvalue_loss_w_l2r"] = qvalue_loss.item()

            self.optimizer_qvalue.zero_grad()
            qvalue_loss.backward()
            self.optimizer_qvalue.step()
        return log

    def update_value(self, states, fixed_qvalues, fixed_log_probs, log):
        for _ in range(self.optim_value_iternum):
            values_pred = self.value(states)
            value_loss = (values_pred - fixed_qvalues + fixed_log_probs).pow(2).mean()
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

    def update_policy(self, states, actions, log):
        log_probs = self.policy.get_log_prob(states, actions)
        actions_mean, actions_log_std = self.policy(states)
        xi = ((actions - actions_mean)/actions_log_std.exp()).detach()
        free_actions = actions_mean + actions_log_std.exp() * xi
        with torch.no_grad():
            square_loss = (self.policy.get_log_prob(states, free_actions) -
                           self.qvalue_net(states, free_actions)).mean()
        policy_loss = log_probs.mean() + square_loss
        log["policy_loss"] = policy_loss.item()

        self.optimizer_policy.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 40)
        self.optimizer_policy.step()
        return log

    def miniupdate(self, batch, log, iter_i, *args, **kwargs):
        states = batch["states"]
        actions = batch["actions"]
        value_targets = batch["value_targets"]

        with torch.no_grad():
            fixed_log_probs = self.policy.get_log_prob(states, actions).data
            fixed_qvalues = self.qvalue_net(states, actions).data

        self.update_value(states, fixed_qvalues, fixed_log_probs, log)
        self.update_qvalue(states, actions, value_targets, log)
        self.update_policy(states, actions, log)

        return log

    def __call__(self, batch, log, iter_i, *args, **kwargs):
        states = batch["states"]
        actions = batch["actions"]
        value_targets = batch["value_targets"]

        with torch.no_grad():
            fixed_log_probs = self.policy.get_log_prob(states, actions).data
            fixed_qvalues = self.qvalue_net(states, actions).data

        num_sample = states.shape[0]
        optim_iter_num = int(np.ceil(num_sample / self.optim_batch_size))

        for _ in range(self.optim_epochs):
            perm = np.arange(num_sample)
            np.random.shuffle(perm)
            perm = np_to_tensor(perm).long()
            if use_gpu and self.cfg["gpu"]:
                perm = perm.cuda()

            # it is important to shuffle samples
            states, actions, value_targets, fixed_qvalues, fixed_log_probs = \
                states[perm], actions[perm], value_targets[perm], fixed_qvalues[perm], fixed_log_probs[perm]

            for i in range(optim_iter_num):
                # do minibatch optimization
                ind = slice(i * self.optim_batch_size, min((i + 1) * self.optim_batch_size, num_sample))
                states_b, actions_b, value_targets_b, fixed_qvalues_b, fixed_log_probs_b = \
                    states[ind], actions[ind], value_targets[ind], fixed_qvalues[ind], fixed_log_probs[ind]

                self.update_qvalue(states_b, actions_b, value_targets_b, log)
                self.update_value(states_b, fixed_qvalues_b, fixed_log_probs_b, log)
                self.update_policy(states_b, actions_b, log)

        return log

    def state_dict(self):
        return {'optimizer_policy': self.optimizer_policy.state_dict(),
                'optimizer_value': self.optimizer_value.state_dict(),
                'optimizer_qvalue': self.optimizer_qvalue.state_dict()}

    def load_state_dict(self, state_dict):
        self.optimizer_policy.load_state_dict(state_dict['optimizer_policy'])
        self.optimizer_value.load_state_dict(state_dict['optimizer_value'])
        self.optimizer_qvalue.load_state_dict(state_dict['optimizer_qvalue'])
