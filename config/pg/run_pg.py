import os
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-env', '-env-name', type=str, default='Swimmer-v2')
    parser.add_argument('--seed', type=int, default=1257)
    FLAGS = parser.parse_args()

    for algo in ['a2c_gym.py', 'ppo_gym.py', 'trpo_gym.py']:
        cmd = 'python config/pg/{} --env-name {} --max-iter-num 2000 --save-model-interval 100 ' \
              '--eval-model-interval 0 --gpu --seed {}'.format(algo, FLAGS.env, FLAGS.seed)
        os.system(cmd)
        print(cmd)

    plot = 'python utils/plot.py --env-name {} --x_len 1000 --save_data'.format(FLAGS.env)
    os.system(plot)
