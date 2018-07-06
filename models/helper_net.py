import math
import torch.nn as nn
import torch.nn.functional as F
from utils.torchs import *


# ====================================================================================== #
# Parent class for all networks
# ====================================================================================== #
class Network(nn.Module):
    def __init__(self, state_dim, conf=None):
        super(Network, self).__init__()
        self.state_dim = state_dim
        self.activate = choose_activate(conf["activate"]) if "activate" in conf else nn.ReLU()

        if isinstance(state_dim, int):
            hidden_size = conf["hidden"] if "hidden" in conf else [128, 128]
            layers = build_fc([state_dim] + hidden_size, activate=self.activate, actlast=True)
            self.last_dim = hidden_size[-1]
        elif isinstance(state_dim, tuple):
            layers = build_conv(state_dim[0])
            layers.append(Flatten())
            conv_out_size = get_out_dim(layers, self.state_dim)
            out_size = conf["out_size"] if "out_size" in conf else 512
            layers += build_fc([conv_out_size, out_size], activate=self.activate, actlast=True)
            self.last_dim = out_size
        else:
            raise TypeError

        self.layers = nn.Sequential(*layers)


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
        raise KeyError('No such activation.'.format(activate_name))


def build_conv(in_channels):
    layers = [
        nn.Conv2d(int(in_channels), 32, kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1),
        nn.ReLU()
    ]

    return layers


def build_fc(dims, activate=nn.ReLU(), actlast=False):
    assert len(dims) >= 2
    layers = [nn.Linear(dims[0], dims[1])]
    for i in range(1, len(dims) - 1):
        if dims[i] and dims[i+1]:
            layers.append(activate)
            layers.append(nn.Linear(dims[i], dims[i+1]))
    if actlast:
        layers.append(activate)

    return layers


def build_noisy_fc(dims, sigma0, activate=nn.ReLU()):
    assert len(dims) >= 2
    layers = [NoisyLinear(dims[0], dims[1], sigma0)]
    for i in range(1, len(dims) - 1):
        if dims[i] and dims[i+1]:
            layers.append(activate)
            layers.append(NoisyLinear(dims[i], dims[i+1], sigma0))

    return layers


# ====================================================================================== #
# Initialization
# ====================================================================================== #
def normalized_columns_init(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
    return out


def weights_init_uniform(m):
    """Initialize the weights uniformly within the interval [−b,b],
    where
        b = sqrt(6 / (f_in + f_out)) for sigmoid units and
        b = 4 * sqrt(6 / (f_in + f_out)) for tangent units.
    """
    if isinstance(m, nn.Conv2d):
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif isinstance(m, nn.Linear):
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


def weights_init_normal(m):
    """Initialize the weights in the way of a zero-mean Gaussian with std,
    where
        std = sqrt(2 / # kernel parameters) for conv map and
        std = sqrt(2 / (f_in + f_out)) for linear map.
    """
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, np.sqrt(2. / n))
    elif isinstance(m, nn.Linear):
        size = m.weight.size()
        m.weight.data.normal_(0, np.sqrt(2. / size[0] + size[1]))
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


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

        noisy_weight = self.noisy_weight * self.noise
        noisy_bias = self.noisy_bias * self.out_noise
        noisy_y = F.linear(x, noisy_weight, noisy_bias)
        return noisy_y + normal_y

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) + ')'
