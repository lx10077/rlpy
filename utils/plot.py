import argparse
import os
import sys
import json
import uuid
import glob
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Force matplotlib to not use any Xwindows backend.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import matplotlib.pyplot as plt
from utils.tools import traindir, assetdir

save_dir = os.path.join(assetdir, 'fig')
os.makedirs(save_dir, exist_ok=True)


def get_reward_from_event(args):
    reward_dict = {}
    base_path = os.path.join(os.path.join(traindir, 'config'), args.env)
    for file in glob.glob(os.path.join(base_path, args.env + '-*')):
        algo = file.split('-')[-1].lower()
        event_file = os.path.join(file, 'train.events')
        events = os.listdir(event_file)
        rewards = {}
        for event in events:
            print('[Info]      Handle {}:{}.'.format(algo, event))
            event_rewards = {}
            try:
                for e in tf.train.summary_iterator(os.path.join(event_file, event)):
                    for v in e.summary.value:
                        if v.tag == 'avg_reward' and e.step not in event_rewards:
                            event_rewards[e.step] = v.simple_value
            except Exception as e:
                print('[Info]      {}.'.format(Exception(e)))
            if len(event_rewards) == 0:
                continue
            rewards.update({event: event_rewards})
        reward_dict.update({algo: rewards})

    if args.save_data:
        out = os.path.join(base_path, args.env + '.data.json')
        json.dump(reward_dict, open(out, 'w'))

    if args.show_info:
        print('+' + '-' * 69 + '+')
        print('|{: ^69}|'.format('Env: ' + args.env))
        print('+' + '-' * 69 + '+')
        print('|{:10s}  {: ^50s} {: >6s}|'.format('Algo', 'Event', 'Len'))
        for key, value in reward_dict.items():
            for k, v in value.items():
                if len(k) > 50:
                    k = k[:50]
                print('|{:10s}  {:50s} {:6d}|'.format(key, k, len(v)))
        print('+' + '-' * 69 + '+')

    return reward_dict


def plot_reward(reward_dict, title, length, dpi=300,
                fig_basename=None, save=True, viz=False):
    try:
        plt.figure(figsize=(6, 6))
    except Exception as e:
        print(Exception(e))

    MEAN_LENGTH = length // 10

    for algo, rewards in reward_dict.items():
        rwds = list(list(rewards.values())[0].values())
        mu = []
        upper = []
        lower = []

        for i in range(min(len(rwds), length)):
            if i < MEAN_LENGTH:
                mean = np.mean(rwds[0:i + 1])
                std = np.std(rwds[0:i + 1])
            else:
                mean = np.mean(rwds[i - MEAN_LENGTH: i])
                std = np.std(rwds[i - MEAN_LENGTH: i])
            mu.append(mean)
            upper.append(mean + 0.5 * std)
            lower.append(mean - 0.5 * std)
        mu, lower, upper = mu[:length], lower[:length], upper[:length]
        plt.plot(np.arange(len(mu)), np.array(mu), linewidth=1.0, label=algo)
        plt.fill_between(np.arange(len(mu)), upper, lower, alpha=0.3)

    plt.grid(True)
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Avg_reward")
    plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
    if save:
        if fig_basename is None:
            fig_basename = title + uuid.uuid4().hex + '.png'
        plt.savefig(os.path.join(save_dir, fig_basename), dpi=dpi)

    if viz:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', '--env-name', type=str, default='Hopper-v2')
    parser.add_argument('--save_data', action='store_true', default=False)
    parser.add_argument('--show_info', action='store_false', default=True)
    parser.add_argument('--x_len', type=int, default=670)
    FLAGS = parser.parse_args()

    r_dict = get_reward_from_event(FLAGS)
    plot_reward(r_dict, FLAGS.env, FLAGS.x_len, )
