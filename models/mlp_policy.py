from models.helper_net import *
from utils.distribution import *


class DiagnormalPolicy(Network):
    def __init__(self, state_dim, action_dim, **kwargs):
        self.conf = {"activate": "tanh",
                     "log_std": 0}
        self.conf.update(kwargs)
        super(DiagnormalPolicy, self).__init__(state_dim, self.conf)

        self.is_disc_action = False
        self.action_dim = action_dim
        self.action_mean = nn.Linear(self.last_dim, action_dim)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)

        if "logstd_on_state" in self.conf and self.conf["logstd_on_state"]:
            self.logstd_on_state = True
            self.action_log_std = nn.Linear(self.last_dim, action_dim)
            self.action_log_std.weight.data.mul_(0.1)
            self.action_log_std.bias.data.mul_(0.0)
        else:
            self.logstd_on_state = False
            self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * self.conf["log_std"])

    def forward(self, x):
        x = self.layers(x)
        action_mean = self.action_mean(x)

        if self.logstd_on_state:
            action_log_std = self.action_log_std(x)
        else:
            action_log_std = self.action_log_std.expand_as(action_mean)

        return action_mean, action_log_std  # Variable

    def select_action(self, x):
        action_mean, action_log_std = self.forward(x)
        action_std = torch.exp(action_log_std)
        action = torch.normal(action_mean, action_std)
        return action.data  # numpy array

    def mean_action(self, x):
        action_mean, _ = self.forward(x)
        return action_mean.data  # numpy array

    def get_kl(self, x, old_stat=None):
        mean1, log_std1 = self.forward(x)

        if old_stat is not None:  # make sure old_stat is detached
            mean0, log_std0 = old_stat
            mean0, log_std0 = mean0.detach(), log_std0.detach()
        else:
            # applied to automatic differentiation in TRPO
            # couldn't be used to calculate the specific KL
            mean0 = Variable(mean1.data).detach()
            log_std0 = Variable(log_std1.data).detach()

        std0, std1 = torch.exp(log_std0), torch.exp(log_std1)
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    def get_log_prob(self, x, actions):
        action_mean, action_log_std = self.forward(x)
        return normal_log_density(actions, action_mean, action_log_std)

    def get_entropy(self, x):
        _, action_log_std = self.forward(x)
        action_std = action_log_std.exp()
        return normal_entropy(action_std)

    def get_fim(self, x):
        mean, _ = self.forward(x)
        cov_inv = self.action_log_std.data.exp().pow(-2).squeeze(0).repeat(x.size(0))
        param_count, std_index, _id, std_id = 0, 0, 0, 0
        for name, param in self.named_parameters():
            if name == "action_log_std":
                std_id = _id
                std_index = param_count
            param_count += param.data.view(-1).shape[0]
            _id += 1
        return cov_inv, mean, {'std_id': std_id, 'std_index': std_index}


class DiscretePolicy(Network):
    def __init__(self, state_dim, action_num, **kwargs):
        self.conf = {"activate": "tanh"}
        self.conf.update(kwargs)
        super(DiscretePolicy, self).__init__(state_dim, self.conf)

        self.is_disc_action = True
        self.action_dim = action_num
        self.action_head = nn.Linear(self.last_dim, action_num)
        self.action_head.weight.data.mul_(0.1)
        self.action_head.bias.data.mul_(0.0)

    def forward(self, x):
        x = self.layers(x)
        action_prob = F.softmax(self.action_head(x), dim=1)
        return action_prob  # return Variable

    def select_action(self, x):
        action_prob = self.forward(x)
        action = action_prob.multinomial().data.numpy()  # numpy array
        return int(action)  # return int

    def mean_action(self, x):
        action_prob = self.forward(x)
        all_action = np_to_var(np.arange(self.action_num)).float().unsqueeze(0)
        action_mean = (all_action * action_prob).sum(1)
        action_mean = action_mean.data.numpy()
        upper = np.ceil(action_mean)
        lower = np.floor(action_mean)
        action = upper if upper + lower > 2 * action_mean else lower  # return the closet numpy array
        return int(action)

    def get_kl(self, x, old_stat=None):
        action_prob1 = self.forward(x)

        if old_stat is not None:  # make sure old_stat is detached
            action_prob0 = old_stat.detach()
        else:
            action_prob0 = Variable(action_prob1.data).detach()

        kl = action_prob0 * (action_prob0.log() - action_prob1.log())
        return kl.sum(1, keepdim=True)

    def get_log_prob(self, x, actions):
        action_prob = self.forward(x)
        return action_prob.gather(1, actions.unsqueeze(1))

    def get_entropy(self, x):
        action_prob = self.forward(x)
        return categorical_entropy(action_prob)

    def get_fim(self, x):
        action_prob = self.forward(x)
        m = action_prob.pow(-1).view(-1).data
        return m, action_prob, {}
