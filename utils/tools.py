import shutil
import os


def info_print(head, msg):
    pre_len = len(head)
    print(('[{:<'+str(pre_len)+'s}]{: <'+str(10-pre_len)+'s}{}').format(head, ' ', msg))


def trainlog_dir(prefix=None):
    if prefix is not None:
        new_dir = os.path.abspath(os.path.join(os.path.dirname(str(prefix)), '../train_log/'))
    else:
        new_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../train_log/'))
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
        print("[Dir]       Create train_log in " + new_dir)
    return new_dir


def set_dir(prefix, name, overwrite=False):
    new_dir = os.path.join(prefix, name)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
        print("[Dir]       Create new_dir in " + new_dir)
    else:
        if overwrite:
            print("[Warning]   Existing new_dir, will verwrite it.")
            shutil.rmtree(new_dir)  # removes all the subdirectories!
            os.makedirs(new_dir)
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
