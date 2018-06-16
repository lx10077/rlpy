from models.helper_net import *


# ====================================================================================== #
# Discriminator used for GAIL
# ====================================================================================== #
class Discriminator(Network):
    def __init__(self, num_inputs, **kwargs):
        self.conf = {"activate": "tanh"}
        self.conf.update(kwargs)
        super(Discriminator, self).__init__(num_inputs, self.conf)
        self.num_inputs = num_inputs

        self.logic = nn.Linear(self.last_dim, 1)
        self.logic.weight.data.mul_(0.1)
        self.logic.bias.data.mul_(0.0)

    def forward(self, x):
        x = self.layers(x)
        probs = F.sigmoid(self.logic(x))
        return probs
