from plotly.graph_objs import Scatter, Line
import numpy as np
import plotly
import os
import time


def assets_dir():
    new_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../assets/'))
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
        print("[Dir] Create asset_dir in " + new_dir)
    return new_dir


def set_dir(prefix, name):
    new_dir = os.path.join(prefix, name)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
        print("[Dir] Create asset_dir in " + new_dir)
    return new_dir


def estimate_advantages(rewards, masks, values, gamma, tau, use_gpu):
    if use_gpu:
        rewards, masks, values = rewards.cpu(), masks.cpu(), values.cpu()
    tensor_type = type(rewards)
    deltas = tensor_type(rewards.size(0), 1)
    advantages = tensor_type(rewards.size(0), 1)

    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        deltas[i] = rewards[i] + gamma * prev_value * masks[i] - values[i]
        advantages[i] = deltas[i] + gamma * tau * prev_advantage * masks[i]

        prev_value = values[i, 0]
        prev_advantage = advantages[i, 0]

    returns = values + advantages
    advantages = (advantages - advantages.mean()) / advantages.std()

    if use_gpu:
        advantages, returns = advantages.cuda(), returns.cuda()
    return advantages, returns


def merge_log(log_list):
    log = dict()
    log['total_reward'] = sum([x['total_reward'] for x in log_list])
    log['num_episodes'] = sum([x['num_episodes'] for x in log_list])
    log['num_steps'] = sum([x['num_steps'] for x in log_list])
    log['avg_reward'] = log['total_reward'] / log['num_episodes']
    log['max_reward'] = max([x['max_reward'] for x in log_list])
    log['min_reward'] = min([x['min_reward'] for x in log_list])
    if 'total_c_reward' in log_list[0]:
        log['total_c_reward'] = sum([x['total_c_reward'] for x in log_list])
        log['avg_c_reward'] = log['total_c_reward'] / log['num_steps']
        log['max_c_reward'] = max([x['max_c_reward'] for x in log_list])
        log['min_c_reward'] = min([x['min_c_reward'] for x in log_list])

    return log


def shuffle_batch(batch, perm):
    for key, value in batch.items():
        batch[key] = value[perm]
    return batch


def get_minibatch(batch, ind):
    minibatch = {}
    for key, value in batch.items():
        minibatch[key] = value[ind]
    return minibatch


# ====================================================================================== #
# Plotting and monitoring
# ====================================================================================== #
def population_plot(xs, ys, title, path):
    """Plots min, max and mean + standard deviation bars of a population over time.

    Parameters
    ----------
    xs: iterations, list or numpy array, shape (N, )
    ys: sum of rewards, list or numpy array, shape (N, num_epsd)
    title: figure title
    path: saving dir
    """
    max_colour, mean_colour, std_colour = 'rgb(0, 132, 180)', 'rgb(0, 172, 237)', 'rgba(29, 202, 255, 0.2)'

    xs, ys = np.array(xs), np.array(ys)
    ys_min, ys_max = ys.min(1).squeeze(), ys.max(1).squeeze()
    ys_mean, ys_std = ys.mean(1).squeeze(), ys.std(1).squeeze()
    ys_upper, ys_lower = ys_mean + ys_std, ys_mean - ys_std

    trace_max = Scatter(x=xs, y=ys_max, line=Line(color=max_colour, dash='dash'), name='Max')
    trace_upper = Scatter(x=xs, y=ys_upper, line=Line(color='transparent'),
                          name='+1 Std. Dev.', showlegend=False)
    trace_mean = Scatter(x=xs, y=ys_mean, fill='tonexty',
                         fillcolor=std_colour, line=Line(color=mean_colour), name='Mean')
    trace_lower = Scatter(x=xs, y=ys_lower, fill='tonexty',
                          fillcolor=std_colour, line=Line(color='transparent'),
                          name='-1 Std. Dev.', showlegend=False)
    trace_min = Scatter(x=xs, y=ys_min, line=Line(color=max_colour, dash='dash'), name='Min')

    plotly.offline.plot({
        'data': [trace_upper, trace_mean, trace_lower, trace_min, trace_max],
        'layout': dict(title=title, xaxis={'title': 'Iteration'}, yaxis={'title': title})
    }, filename=os.path.join(path, title + '.html'), auto_open=False)


def log_plot(writer, update_log, i_iter):
    """Monitor training process by TensorboardX.

    Parameters
    ----------
    writer: SummaryWriter in TensorboardX, run "tensorboard --logdir runs".
    update_log: dict like {tag: value}, containing all sampling and updating information.
    i_iter: int, global step.
    """

    reward_dict = {}
    action_dict = {}
    if not update_log:
        print("[Warning] Empty updating log!")
        return

    for tag, value in update_log.items():
        if "reward" in tag:
            reward_dict[tag] = value
        elif "action" in tag:
            action_dict[tag] = value
        else:
            writer.add_scalar(tag, value, i_iter)

    if 'total_reward' in reward_dict.keys():
        del reward_dict['total_reward']
    writer.add_scalars('reward', reward_dict, i_iter)
