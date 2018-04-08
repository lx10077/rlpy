class Evaluator(object):
    def __init__(self, agent, cfg):
        self.agent = agent
        self.cfg = cfg

    def test(self):
        raise NotImplementedError

    def _test(self):
        raise NotImplementedError

    def plot(self):
        raise NotImplementedError

    def monitor(self):
        raise NotImplementedError
