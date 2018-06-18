import torch
import numpy as np
from torch.autograd import Variable


# ====================================================================================== #
# Pytorch operation shortcuts
# ====================================================================================== #
use_gpu = torch.cuda.is_available()
DoubleTensor = torch.DoubleTensor
FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor
ByteTensor = torch.ByteTensor


def cuda_var(tensor, gpu=False, *args, **kwargs):
    if use_gpu and gpu:
        return torch.autograd.Variable(tensor, *args, **kwargs).cuda()
    else:
        return torch.autograd.Variable(tensor, *args, **kwargs)


def ones(*shape, gpu=False):
    return torch.ones(*shape).cuda() if use_gpu and gpu else torch.ones(*shape)


def zeros(*shape, gpu=False):
    return torch.zeros(*shape).cuda() if use_gpu and gpu else torch.zeros(*shape)


def one_hot(x, n):
    is_var = False
    if isinstance(x, Variable):
        x = x.data
        is_var = True
    assert x.dim() == 2, "Incompatible dim {:d} for input. Dim must be 2.".format(x.dim())
    one_hot_x = torch.zeros(x.size(0), n)
    one_hot_x.scatter_(1, x, 1)
    return one_hot_x if not is_var else cuda_var(one_hot_x)


def np_to_tensor(nparray):
    assert isinstance(nparray, np.ndarray)
    return torch.from_numpy(nparray)


def np_to_var(nparray, gpu=False, *args, **kwargs):
    assert isinstance(nparray, np.ndarray)
    return_var = Variable(torch.from_numpy(nparray), *args, **kwargs)
    return return_var.cuda() if gpu else return_var


def np_to_cuda_var(nparray, *args, **kwargs):
    assert isinstance(nparray, np.ndarray)
    return Variable(torch.from_numpy(nparray), *args, **kwargs).cuda()


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# ====================================================================================== #
# Getting and setting operations
# ====================================================================================== #
def set_seed(seed, env=None):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if env is not None:
        env.seed(seed)


def get_gym_info(env_factory):
    import gym
    from gym.spaces.box import Box
    print('[Prepare]   Get env information...')
    try:
        env = env_factory(0)
    except TypeError:
        if isinstance(env_factory, gym.wrappers.time_limit.TimeLimit):
            env = env_factory
        else:
            raise TypeError('Input should be env or env_factory.')

    state_dim = env.observation_space.shape
    if len(state_dim) == 1:  # If state is 3-dim image.
        state_dim = state_dim[0]

    if isinstance(env.action_space, Box):
        # If action space is continous, return action dimension.
        action_dim = env.action_space.shape[0]
        is_disc_action = False
    else:
        # If action space is discrete, return number of actions,
        # but still called action_dim
        action_dim = env.action_space.n
        is_disc_action = True

    env.close()
    del env
    return state_dim, action_dim, is_disc_action


def get_state_dict(file):
    try:
        pretrain_state_dict = torch.load(file)
    except AssertionError:
        pretrain_state_dict = torch.load(file, map_location=lambda storage, location: storage)
    return pretrain_state_dict


def get_out_dim(module, indim):
    if isinstance(module, list):
        module = torch.nn.Sequential(*module)
    fake_input = Variable(torch.zeros(indim).unsqueeze(0))
    output_size = module(fake_input).view(-1).size()[0]
    return output_size


def get_flat_params_from(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))

    flat_params = torch.cat(params)
    return flat_params


def set_flat_params_to(model, flat_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size


def assign_params(from_model, to_model):
    if from_model is not None and to_model is not None:
        params = get_flat_params_from(from_model)
        set_flat_params_to(to_model, params)
    return


def get_flat_grad_from(inputs, grad_grad=False):
    grads = []
    for param in inputs:
        if grad_grad:
            grads.append(param.grad.grad.view(-1))
        else:
            if param.grad is None:
                grads.append(Variable(zeros(param.data.view(-1).shape)))
            else:
                grads.append(param.grad.view(-1))

    flat_grad = torch.cat(grads)
    return flat_grad


# ====================================================================================== #
# Computing and estimating operations
# ====================================================================================== #
def compute_flat_grad(output, inputs, filter_input_ids=set(), retain_graph=False, create_graph=False):
    if create_graph:
        retain_graph = True

    inputs = list(inputs)
    params = []
    for i, param in enumerate(inputs):
        if i not in filter_input_ids:
            params.append(param)

    grads = torch.autograd.grad(output, params, retain_graph=retain_graph, create_graph=create_graph)

    j = 0
    out_grads = []
    for i, param in enumerate(inputs):
        if i in filter_input_ids:
            out_grads.append(Variable(zeros(param.data.view(-1).shape)))
        else:
            out_grads.append(grads[j].view(-1))
            j += 1
    grads = torch.cat(out_grads)

    for param in params:
        param.grad = None
    return grads
