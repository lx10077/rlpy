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


class Discriminator(nn.Module):
    def __init__(self, num_inputs, hidden_size=(128, 128), activation='tanh'):
        super(Discriminator, self).__init__()
        if activation == 'tanh':
            self.activation = F.tanh
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'sigmoid':
            self.activation = F.sigmoid

        self.affine_layers = nn.ModuleList()
        last_dim = num_inputs
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        self.logic = nn.Linear(last_dim, 1)
        self.logic.weight.data.mul_(0.1)
        self.logic.bias.data.mul_(0.0)

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))

        prob = F.sigmoid(self.logic(x))
        return prob
