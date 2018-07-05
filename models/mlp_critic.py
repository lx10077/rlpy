from models.helper_net import *


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


class LinearVariate(Network):
    def __init__(self, state_dim, action_dim, **kwargs):
        assert isinstance(state_dim, int)
        self.conf = {"activate": "relu",
                     "hidden": [100],
                     "log_std": 0}
        self.conf.update(kwargs)
        super(LinearVariate, self).__init__(state_dim, self.conf)
        self.action_dim = action_dim

        self.action_mean = nn.Linear(self.last_dim, action_dim)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)

    def forward(self, x, a, qw):
        x = self.layers(x)
        action_mean = self.action_mean(x)
        variate = qw * (a - action_mean)
        return variate.sum(1, keepdim=True)


class QuadraticVariate(Network):
    def __init__(self, state_dim, action_dim, **kwargs):
        assert isinstance(state_dim, int)
        self.conf = {"activate": "relu",
                     "hidden": [100],
                     "log_std": 0}
        self.conf.update(kwargs)
        super(QuadraticVariate, self).__init__(state_dim, self.conf)
        self.action_dim = action_dim

        self.action_mean = nn.Linear(self.last_dim, action_dim)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)
        self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * self.conf["log_std"])

    def forward(self, x, a):
        x = self.layers(x)
        action_mean = self.action_mean(x)
        action_log_std = self.action_log_std.expand_as(action_mean)
        assert a.shape == action_mean.shape == action_log_std.shape
        variate = (a - action_mean) / action_log_std.exp()
        return -variate.pow(2).sum(1, keepdim=True)


class MlpVariate(Network):
    def __init__(self, state_dim, action_dim, **kwargs):
        assert isinstance(state_dim, int)
        self.conf = {"activate": "relu",
                     "hidden": [100, None, 100+action_dim, 100],
                     "layerlst": True,
                     "log_std": 0}
        self.conf.update(kwargs)
        super(MlpVariate, self).__init__(state_dim, self.conf)
        self.action_dim = action_dim

        self.variate = nn.Linear(self.last_dim, 1)
        self.variate.weight.data.mul_(0.1)
        self.variate.bias.data.mul_(0.0)

    def forward(self, x, a):
        for module in self.layers[:2]:
            x = module(x)
        assert x.dim() == a.dim() and x.shape[0] == a.shape[0]
        x = torch.cat([x, a], dim=1)
        for module in self.layers[2:]:
            x = module(x)
        return self.variate(x)


# ====================================================================================== #
# DQN network and its variants
# ====================================================================================== #
class BasicDqn(nn.Module):
    def __init__(self, state_dim, action_dim, epsilon=0):
        super(BasicDqn, self).__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.epsilon = epsilon
        self.conv = build_conv(state_dim[0])

        conv_out_size = get_out_dim(self.conv, self.state_dim)
        self.fc = build_fc([conv_out_size, 512, self.action_dim])
        self.apply(weights_init_normal)

    def forward(self, x):
        fx = x.float()
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.fc(conv_out)

    def select_action(self, x):
        if np.random.uniform() <= self.epsilon:
            actions = np.random.randint(0, self.action_dim)
        else:
            q_vals = self.forward(x)
            q_vals = q_vals.data.view(1, -1)
            actions = q_vals.numpy().argmax()
        return actions  # int


class DuelingDqn(nn.Module):
    def __init__(self, state_dim, action_dim,
                 epsilon=0, noisy=False, sigma0=None):
        super(DuelingDqn, self).__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.is_disc_action = True
        self.noisy = noisy
        self.sigma0 = sigma0
        self.epsilon = epsilon
        self.conv = build_conv(state_dim[0])

        conv_out_dim = get_out_dim(self.conv, self.state_dim)
        adv_dims = [conv_out_dim, 512, self.action_dim]
        val_dims = [conv_out_dim, 512, 1]

        if self.noisy and self.sigma0:
            self.adv = build_noisy_fc(adv_dims, sigma0)
            self.val = build_noisy_fc(val_dims, sigma0)
        else:
            self.adv = build_fc(adv_dims)
            self.val = build_fc(val_dims)
        self.apply(weights_init_normal)

    def forward(self, x):
        fx = x.float()
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        adv = self.adv(conv_out)
        val = self.val(conv_out)
        return val - adv.mean(1, keepdim=True) + adv

    def select_action(self, x):
        if np.random.uniform() <= self.epsilon:
            actions = np.random.randint(0, self.action_dim)
        else:
            q_vals = self.forward(x)
            q_vals = q_vals.data.view(1, -1)
            actions = q_vals.numpy().argmax()
        return actions  # int


class DistDqn(nn.Module):
    def __init__(self, state_dim, action_dim, n_atoms=51, vmin=-10, vmax=10,
                 epsilon=0, noisy=False, sigma0=None):
        super(DistDqn, self).__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.is_disc_action = True
        self.n_atoms = n_atoms
        self.vmin = vmin
        self.vmax = vmax
        self.noisy = noisy
        self.epsilon = epsilon
        self.conv = build_conv(state_dim[0])

        self.delta_z = (self.vmax - self.vmin) / (self.n_atoms - 1)
        self.z_vals = np_to_tensor(np.linspace(self.vmin, self.vmax, self.n_atoms).astype(np.float32))

        conv_out_size = get_out_dim(self.conv, self.state_dim)
        fc_dims = [conv_out_size, 512, self.action_dim * self.n_atoms]
        if self.noisy and self.sigma0:
            self.fc = build_noisy_fc(fc_dims, sigma0)
        else:
            self.fc = build_fc(fc_dims)
        self.apply(weights_init_normal)

    def forward(self, x):
        fx = x.float()
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        logits = self.fc(conv_out).view(fx.size()[0], self.action_dim, self.n_atoms)
        probs = F.softmax(logits, 2)
        return probs

    def compute_q_values(self, x):
        probs = self.forward(x)
        q_vals = (probs * self.z_vals).sum(2)
        return q_vals.data, probs.data

    def select_action(self, x):
        if np.random.uniform() <= self.epsilon:
            actions = np.random.randint(0, self.action_dim)
        else:
            q_vals, _ = self.compute_q_values(x)
            q_vals = q_vals.view(1, -1)
            actions = q_vals.numpy().argmax()
        return actions  # int


# ====================================================================================== #
# Long short-term memory for A3C
# ====================================================================================== #
class A3cLstm(torch.nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(A3cLstm, self).__init__()
        if isinstance(num_inputs, tuple) and len(num_inputs) == 3:
            num_inputs = num_inputs[0]

        self.conv1 = nn.Conv2d(num_inputs, 32, 5, stride=1, padding=2)
        self.maxp1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 5, stride=1, padding=1)
        self.maxp2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=1, padding=1)
        self.maxp3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.maxp4 = nn.MaxPool2d(2, 2)

        self.lstm = nn.LSTMCell(1024, 512)
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, num_actions)

        self.apply(weights_init_uniform)
        relu_gain = nn.init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)
        self.conv4.weight.data.mul_(relu_gain)

        self.actor_linear.weight.data = normalized_columns_init(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)

        self.critic_linear.weight.data = normalized_columns_init(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.hx, self.cx = zeros(1, 512), zeros(1, 512)

        self.train()

    def forward(self, inputs):
        x = F.relu(self.maxp1(self.conv1(inputs)))
        x = F.relu(self.maxp2(self.conv2(x)))
        x = F.relu(self.maxp3(self.conv3(x)))
        x = F.relu(self.maxp4(self.conv4(x)))
        x = x.view(x.size(0), -1)
        self.hx, self.cx = self.lstm(x, (self.hx, self.cx))

        value = self.critic_linear(self.hx)
        logit = self.actor_linear(self.hx)
        prob = F.softmax(logit, dim=1)

        return value, prob

    def reset_hc(self, done, gpu=False, gpu_id=None):
        if done:
            self.hx, self.cx = zeros(1, 512), zeros(1, 512)
        else:
            self.hx, self.cx = self.hx.detach(), self.cx.detach()
        if gpu:
            self.hx, self.cx = self.hx.cuda(gpu_id), self.cx.cuda(gpu_id)
