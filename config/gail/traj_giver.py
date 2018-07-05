from models.mlp_policy import DiagnormalPolicy, DiscretePolicy
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import RandomSampler
from itertools import count
from utils import *
import gym
import pickle
import glob


class ExpertDataset(Dataset):
    def __init__(self, expert_traj, batch_size):
        self.expert_traj = expert_traj
        self.batch_size = batch_size

    def __len__(self):
        return self.expert_traj.shape[0]

    def __getitem__(self, index):
        return np_to_tensor(self.expert_traj[index])


def get_expert_loader(batch_size, expert_traj=None, expert_path=None):
    if expert_traj is not None:
        expert_ds = ExpertDataset(expert_traj, batch_size)
    elif expert_path is not None:
        try:
            with open(expert_path, 'rb') as f:
                expert_traj = pickle.load(f)
        except Exception as e:
            raise Exception(e)
        expert_ds = ExpertDataset(expert_traj, batch_size)
    else:
        raise ValueError('No input.')
    expert_dl = DataLoader(expert_ds,
                           batch_size=batch_size,
                           sampler=RandomSampler(expert_ds),
                           drop_last=True)
    return expert_dl


class TrajGiver(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.batch_size = 50000
        self.max_expert_state_num = cfg['max_expert_state_num'] if 'max_expert_state_num' in cfg else 200000

    def __call__(self, expert_path=None, prefer='clipppo', threshold=0):
        possible_traj_dir = os.path.join(assetdir, 'expert_traj/{}-expert-traj.p'.format(self.cfg['env_name']))
        if os.path.exists(possible_traj_dir):
            print('[Info]      Find export data in {}.'.format(possible_traj_dir))
            return get_expert_loader(self.batch_size, expert_path=possible_traj_dir)
        else:
            env, running_state, policy_net = self.set_policy()
            path = self.find_expert(expert_path, prefer=prefer)
            running_state, policy_net = self.load_expert(path, running_state, policy_net)
            traj = self.make_traj(env, policy_net, running_state, threshold=threshold)
            self.save_traj(traj)
            return get_expert_loader(self.batch_size, expert_traj=traj)

    def set_policy(self):
        env = gym.make(self.cfg['env_name'])
        set_seed(self.cfg['seed'])
        torch.set_default_tensor_type('torch.DoubleTensor')
        state_dim, action_dim, is_disc_action = get_gym_info(env)
        running_state = ZFilter((state_dim,), clip=5)

        if is_disc_action:
            policy_net = DiscretePolicy(state_dim, action_dim)
        else:
            policy_net = DiagnormalPolicy(state_dim, action_dim, log_std=self.cfg['log_std'])
        return env, running_state, policy_net

    def find_expert(self, expert_path, prefer='clipppo'):
        if expert_path is not None:
            return expert_path
        else:
            possible_expert = {}
            for file_dir in glob.glob(os.path.join(traindir, 'config/' + self.cfg['env_name'] + '/*')):
                full_name = os.path.basename(file_dir)
                config_name = '-'.join(full_name.split('-')[:2])
                if config_name == self.cfg['env_name']:
                    possible_expert[full_name] = file_dir

            if len(possible_expert) == 0:
                return None

            for name in possible_expert.keys():
                if prefer in name.lower():
                    candidate = possible_expert[name]
                    break
            else:
                # TO DO: relatively better export can be finded based on trainlog.txt
                candidate = np.random.choice(list(possible_expert.values()))
            print('Find export in {}.'.format(candidate))
            return os.path.join(candidate, 'models/latest')

    def load_expert(self, path, running_state, policy_net):
        if path is None:
            raise ValueError('Expert policy path is None. NO EXPERT.')
        save_dict = get_state_dict(path)
        policy_net.load_state_dict(save_dict['nets']['policy'])

        if "running_state" in save_dict:
            running_state.load_state_dict(save_dict["running_state"])
            return running_state, policy_net
        else:
            return None, policy_net

    def make_traj(self, env, policy_net, running_state=None, render=False, threshold=0):
        num_steps = 0
        expert_traj = []
        for i_episode in count():
            state = env.reset()
            if running_state is not None:
                state = running_state(state, False)
            episode_reward, episode_step = 0, 0
            episode_traj = []

            for episode_step in range(10000):
                state_var = np_to_tensor(state).unsqueeze(0)
                # choose mean action
                action = policy_net(state_var)[0].cpu().numpy()[0]
                # choose stochastic action
                # action = policy_net.select_action(state_var)[0].cpu().numpy()
                action = int(action) if policy_net.is_disc_action else action.astype(np.float64)
                next_state, reward, done, _ = env.step(action)

                if running_state is not None:
                    next_state = running_state(next_state, False)

                episode_reward += reward
                num_steps += 1

                episode_traj.append(np.hstack([state, action]))

                if render:
                    env.render()
                if done or num_steps >= self.max_expert_state_num:
                    break

                state = next_state

            info_print('Info', 'Episode {} Steps {} \t reward: {:.2f}\t num step: {}'.format(
                i_episode, episode_step, episode_reward, num_steps)
                       )

            if threshold == 0:
                threshold = episode_reward * 0.95

            if episode_reward >= threshold:
                expert_traj.extend(episode_traj)
            else:
                info_print('Info', 'Discard it...')
                num_steps -= (episode_step + 1)

            if num_steps >= self.max_expert_state_num:
                break
        env.close()
        del env
        return np.stack(expert_traj)

    def save_traj(self, expert_traj):
        info_print('Save', 'Saving trajectories...')
        set_dir(assetdir, 'expert_traj')
        save_path = os.path.join(assetdir, 'expert_traj/{}-expert-traj.p'.format(self.cfg['env_name']))
        with open(save_path, 'wb') as f:
            pickle.dump(expert_traj, f)
