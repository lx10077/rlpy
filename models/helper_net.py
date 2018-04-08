import torch.nn.functional as F
import math
from utils.torch import *


# ====================================================================================== #
# Parent class for all networks
# ====================================================================================== #
class Network(nn.Module):
    def __init__(self, state_dim, conf=None):
        super(Network, self).__init__()
        self.state_dim = state_dim
        self.activate = choose_activate(conf["activate"]) if "activate" in conf else nn.ReLU()
        self.returnlayerlst = conf["layerlst"] if "layerlst" in conf else False

        if isinstance(state_dim, int):
            hidden_size = conf["hidden"] if "hidden" in conf else [128, 128]
            layers = build_fc([state_dim] + hidden_size, returnlst=True,
                              activate=self.activate, actlast=True)
            self.last_dim = hidden_size[-1]
        elif isinstance(state_dim, tuple):
            layers = build_conv(state_dim[0], returnlst=True)
            layers.append(Flatten())
            conv_out_size = get_out_dim(layers, self.state_dim)
            out_size = conf["out_size"] if "out_size" in conf else 512
            layers += build_fc([conv_out_size, out_size], returnlst=True,
                               activate=self.activate, actlast=True)
            self.last_dim = out_size
        else:
            raise TypeError

        self.layers = layers if self.returnlayerlst else nn.Sequential(*layers)


# ====================================================================================== #
# Network building functions
# ====================================================================================== #
def choose_activate(activate_name):
    if activate_name == "relu":
        return nn.ReLU()
    elif activate_name == "tanh":
        return nn.Tanh()
    elif activate_name == "leakyrelu":
        return nn.LeakyReLU()
    elif activate_name == "selu":
        return nn.SELU()
    elif activate_name == "sigmoid":
        return nn.Sigmoid()
    elif activate_name == "elu":
        return nn.ELU()
    else:
        raise KeyError


def build_conv(in_channels, returnlst=False):
    layers = [
        nn.Conv2d(int(in_channels), 32, kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1),
        nn.ReLU()
    ]

    return layers if returnlst else nn.Sequential(*layers)


def build_fc(dims, returnlst=False, activate=nn.ReLU(), actlast=False):
    assert len(dims) >= 2
    layers = [nn.Linear(dims[0], dims[1])]
    for i in range(1, len(dims) - 1):
        if dims[i] and dims[i+1]:
            layers.append(activate)
            layers.append(nn.Linear(dims[i], dims[i+1]))
    if actlast:
        layers.append(activate)

    return layers if returnlst else nn.Sequential(*layers)


def build_noisy_fc(dims, sigma0, returnlst=False, activate=nn.ReLU()):
    assert len(dims) >= 2
    layers = [NoisyLinear(dims[0], dims[1], sigma0)]
    for i in range(1, len(dims) - 1):
        if dims[i] and dims[i+1]:
            layers.append(activate)
            layers.append(NoisyLinear(dims[i], dims[i+1], sigma0))

    return layers if returnlst else nn.Sequential(*layers)


# ====================================================================================== #
# Useful designed layers
# ====================================================================================== #
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size()[0], -1)


# ====================================================================================== #
# Factorised Gaussian NoisyNet
# ====================================================================================== #
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma0):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.noisy_weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.noisy_bias = nn.Parameter(torch.zeros(out_features))
        self.reset_parameters()

        self.noise_std = sigma0 / math.sqrt(self.in_features)
        self.in_noise = torch.zeros(in_features).float()
        self.out_noise = torch.zeros(out_features).float()
        self.noise = None
        self.sample_noise()

    def sample_noise(self):
        self.in_noise.normal_(0, self.noise_std)
        self.out_noise.normal_(0, self.noise_std)
        self.noise = torch.mm(self.out_noise.view(-1, 1), self.in_noise.view(1, -1))

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.noisy_weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
            self.noisy_bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        normal_y = F.linear(x, self.weight, self.bias)
        if not x.volatile:
            # update the noise once per update
            self.sample_noise()

        noisy_weight = self.noisy_weight * Variable(self.noise)
        noisy_bias = self.noisy_bias * Variable(self.out_noise)
        noisy_y = F.linear(x, noisy_weight, noisy_bias)
        return noisy_y + normal_y

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) + ')'
