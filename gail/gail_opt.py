from core.ppo import ClipPpoUpdater
from core.trpo import TrpoUpdater
from torch.utils.data import DataLoader
from utils.torchs import *


class GailUpdater(object):
    def __init__(self, nets, optimizers, cfg, suboptimizer='ppo'):
        self.policy = nets['policy']
        self.value = nets['value']
        self.discrim = nets['discrim']
        self.optimizer_policy = optimizers['optimizer_policy']
        self.optimizer_value = optimizers['optimizer_value']
        self.optimizer_discrim = optimizers['optimizer_discrim']
        if use_gpu and cfg['gpu']:
            self.discrim_criterion = torch.nn.BCELoss().cuda()
        else:
            self.discrim_criterion = torch.nn.BCELoss()

        self.expert_dl = None
        self.gpu = cfg['gpu'] if 'gpu' in cfg else False
        self.optim_discrim_iternum = cfg['optim_discrim_iternum'] if 'noptim_discrim_iternum' in cfg else 1

        if suboptimizer.lower() == 'ppo':
            self.suboptimizer = ClipPpoUpdater(self.policy, self.value,
                                               self.optimizer_policy, self.optimizer_value, cfg)
        elif suboptimizer.lower() == 'trpo':
            self.suboptimizer = TrpoUpdater(self.policy, self.value, cfg, use_fim=False)
        else:
            raise TypeError('{} is NOT surpported.'.format(suboptimizer))

    def load_traj(self, expert_dl):
        assert isinstance(expert_dl, DataLoader)
        self.expert_dl = expert_dl

    def __call__(self, batch, log, *args, **kwargs):
        log = self.suboptimizer(batch, log, *args, **kwargs)

        states = Variable(batch["states"])
        actions = Variable(batch["actions"])
        expert_traj = Variable(iter(self.expert_dl).next())
        if use_gpu and self.gpu:
            expert_traj = expert_traj.cuda()

        valid = Variable(ones((states.shape[0], 1), self.gpu and use_gpu))
        fake = Variable(zeros((expert_traj.shape[0], 1), self.gpu and use_gpu))

        for _ in range(self.optim_discrim_iternum):

            # Sample a minibatch of expert trajectories
            gen_o = self.discrim(torch.cat([states, actions], 1))
            expert_o = self.discrim(expert_traj)

            discrim_loss = self.discrim_criterion(expert_o, fake) + self.discrim_criterion(gen_o, valid)
            log["discrim_loss"] = discrim_loss.data[0]

            self.optimizer_discrim.zero_grad()
            discrim_loss.backward()
            self.optimizer_discrim.step()
        return log

    def state_dict(self):
        state_dict = self.suboptimizer.state_dict()
        state_dict['optimizer_discrim'] = self.optimizer_discrim.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        self.optimizer_discrim.load_state_dict(state_dict['optimizer_discrim'])
        del state_dict['optimizer_discrim']
        self.suboptimizer.load_state_dict(state_dict)
