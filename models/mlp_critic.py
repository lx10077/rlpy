from models.helper_net import *


class Value(nn.Module):
    def __init__(self, state_dim, hidden_size=(128, 128), activation='tanh'):
        super().__init__()
        if activation == 'tanh':
            self.activation = F.tanh
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'sigmoid':
            self.activation = F.sigmoid

        self.affine_layers = nn.ModuleList()
        last_dim = state_dim
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        self.value_head = nn.Linear(last_dim, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))

        value = self.value_head(x)
        return value


# ====================================================================================== #
# Baseline functions
# ====================================================================================== #
class ValueFunction(Network):
    def __init__(self, state_dim, **kwargs):
        self.conf = {"activate": "tanh"}
        self.conf.update(kwargs)
        super(ValueFunction, self).__init__(state_dim, self.conf)

        self.value_head = nn.Linear(self.last_dim, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

    def forward(self, x):
        x = self.layers(x)
        value = self.value_head(x)
        return value
