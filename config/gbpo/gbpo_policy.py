from models.mlp_policy import DiagnormalPolicy
from collections import deque
from utils.torchs import set_flat_params_to
import torch


class AdditiveDiagnormalPolicy(DiagnormalPolicy):
    def __init__(self, state_dim, action_dim, policy_num, alpha, gpu=False, **kwargs):
        super(AdditiveDiagnormalPolicy, self).__init__(state_dim, action_dim, **kwargs)
        self.policy_num = policy_num
        assert policy_num >= 1
        self.policy_type = DiagnormalPolicy
        self.alpha = alpha
        self.gpu = gpu
        self.update_time = 0
        self.policies = deque(maxlen=self.policy_num)
        new_policy = self.policy_type(self.state_dim, self.action_dim, **self.conf)
        self.policies.append(new_policy)

    @property
    def is_full(self):
        return len(self.policies) == self.policy_num

    def add_policy(self, new_flat_param=None):
        new_policy = self.generate_policy(new_flat_param)
        self.policies.append(new_policy)

    def clear_policy(self, new_flat_param):
        self.policies = deque(maxlen=self.policy_num)
        self.add_policy(new_flat_param)

    def generate_policy(self, flat_param=None):
        policy = self.policy_type(self.state_dim, self.action_dim, **self.conf)
        if self.gpu and torch.cuda.is_available():
            policy = policy.to(torch.device("cuda"))
        if flat_param is not None:
            set_flat_params_to(policy, flat_param)
        return policy

    def forward(self, x):
        with torch.no_grad():
            action_mean, action_log_std = self.policies[-1](x)
            for i in range(-2, -len(self.policies)-1, -1):
                policy = self.policies[i]
                action_mean = self.alpha * action_mean + (1 - self.alpha) * policy(x)[0]
                action_log_std = self.alpha * action_mean + (1 - self.alpha) * policy(x)[1]
            return action_mean, action_log_std

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dicts = {}
        for i, policy in enumerate(self.policies):
            state_dicts['policy{}'.format(i)] = policy.state_dict(destination, prefix, keep_vars)
        state_dicts.update({'update_time': self.update_time})
        state_dicts.update({'policy_type': 'DiagnormalPolicy'})
        state_dicts.update({'policy_num': self.policy_num})
        return state_dicts

    def load_state_dict(self, state_dict, strict=True):
        assert 'policy_type' in state_dict and state_dict['policy_type'] == 'DiagnormalPolicy'
        self.policy_num = state_dict['policy_num']
        self.update_time = state_dict['update_time']
        del state_dict['policy_type']
        del state_dict['policy_num']
        del state_dict['update_time']

        self.policies = deque(maxlen=self.policy_num)
        for key in sorted(state_dict.keys()):
            model = self.generate_policy()
            model.load_state_dict(state_dict[key])
            self.policies.append(model)

    def to(self, device):
        if device.type == 'cuda':
            for i in range(len(self.policies)):
                self.policies[i] = self.policies[i].to(torch.device('cuda'))
        elif device.type == 'cpu':
            for i in range(len(self.policies)):
                self.policies[i] = self.policies[i].to(torch.device('cpu'))
        return self
