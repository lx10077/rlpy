from core.ppo import ClipPpoUpdater
from core.trpo import TrpoUpdater
from utils.torchs import *


class GailUpdater(object):
    def __init__(self, nets, optimizers, cfg, suboptimizer='ppo'):
        self.policy = nets['policy']
        self.value = nets['value']
        self.discrim = nets['discrim']
        self.optimizer_policy = optimizers['optimizer_policy']
        self.optimizer_value = optimizers['optimizer_value']
        self.optimizer_discrim = optimizers['optimizer_discrim']
        self.discrim_criterion = torch.nn.BCELoss()
        self.expert_traj = None
        self.gpu = cfg['gpu'] if 'gpu' in cfg else False

        if suboptimizer.lower() == 'ppo':
            self.suboptimizer = ClipPpoUpdater(self.policy, self.value,
                                               self.optimizer_policy, self.optimizer_value, cfg)
        elif suboptimizer.lower() == 'trpo':
            self.suboptimizer = TrpoUpdater(self.policy, self.value, cfg, use_fim=False)
        else:
            raise TypeError('{} is NOT surpported.'.format(suboptimizer))

    def set_traj(self, expert_traj):
        if isinstance(expert_traj, list):
            expert_traj = np.array(expert_traj)
        self.expert_traj = np_to_var(expert_traj)
        if torch.cuda.is_available() and self.gpu:
            self.expert_traj = self.expert_traj.cuda()

    def __call__(self, batch, log, *args, **kwargs):
        log = self.suboptimizer(batch, log, *args, **kwargs)

        states = Variable(batch["states"])
        actions = Variable(batch["actions"])
        for _ in range(3):
            g_o = self.discrim(torch.cat([states, actions], 1))
            e_o = self.discrim(self.expert_traj)

            loss1 = self.discrim_criterion(g_o, Variable(ones((states.shape[0], 1), self.gpu)))
            loss2 = self.discrim_criterion(e_o, Variable(zeros((self.expert_traj.shape[0], 1), self.gpu)))
            discrim_loss = loss1 + loss2
            log["discrim_loss"] = discrim_loss.data[0]

            self.optimizer_discrim.zero_grad()
            discrim_loss.backward()
            self.optimizer_discrim.step()
        return log

    def state_dict(self):
        return self.suboptimizer.state_dict().update(
            {'optimizer_discrim': self.optimizer_discrim.state_dict()}
        )

    def load_state_dict(self, state_dict):
        self.optimizer_discrim.load_state_dict(state_dict['optimizer_discrim'])
        del state_dict['optimizer_discrim']
        self.suboptimizer.load_state_dict(state_dict)
