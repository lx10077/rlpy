


class GailUpdater(object):
    def __init__(self, nets, optimizers, cfg):
        self.policy = nets['policy']
        self.value = nets['value']
        self.discrim = nets['discrim']
        self.optimizer_policy = optimizers['optimizer_policy']
        self.optimizer_value = optimizers['optimizer_value']
        self.optimizer_discrim = optimizers['optimizer_discrim']
        self.cfg = cfg
