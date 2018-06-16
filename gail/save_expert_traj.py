import argparse
import gym
import os
import sys
import pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from itertools import count
from utils import *
from torch.autograd import Variable

Tensor = DoubleTensor
torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='Save expert trajectory')
parser.add_argument('--env-name', default="Hopper-v1", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--model-path', metavar='G',
                    help='name of the expert model')
parser.add_argument('--render', action='store_true', default=False,
                    help='render the environment')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--max-expert-state-num', type=int, default=50000, metavar='N',
                    help='maximal number of main iterations (default: 50000)')
args = parser.parse_args()

env = gym.make(args.env_name)
set_seed(args.seed)
state_dim, action_dim, is_disc_action = get_gym_info(env)

try:
    policy_net, _, running_state = pickle.load(open(args.model_path, "rb"))
except Exception as e:
    info_print('Error', 'Fail to load saved model and running states.')
    raise Exception(e)
expert_traj = []


def main_loop():

    num_steps = 0
    for i_episode in count():

        state = env.reset()
        state = running_state(state)
        reward_episode = 0

        for _ in range(10000):
            state_var = Variable(np_to_tensor(state).unsqueeze(0),  volatile=True)
            # choose mean action
            action = policy_net(state_var)[0].data[0].cpu().numpy()
            # choose stochastic action
            # action = policy_net.select_action(state_var)[0].cpu().numpy()
            action = int(action) if is_disc_action else action.astype(np.float64)
            next_state, reward, done, _ = env.step(action)
            next_state = running_state(next_state)
            reward_episode += reward
            num_steps += 1

            expert_traj.append(np.hstack([state, action]))

            if args.render:
                env.render()
            if done or num_steps >= args.max_expert_state_num:
                break

            state = next_state

        info_print('Save traj', 'Episode {}\t reward: {:.2f}'.format(i_episode, reward_episode))

        if num_steps >= args.max_expert_state_num:
            break


if __name__ == "__main__":
    main_loop()
    save_dict = {"traj": np.stack(expert_traj),
                 "running state": running_state}
    save_path = os.path.join(asset_dir, 'expert_traj/{}_expert_traj.p'.format(args.env_name))
    with open(save_path, 'wb') as f:
        pickle.dump(save_dict, f)
