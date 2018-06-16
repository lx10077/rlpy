import argparse
import gym
import os
import sys
import pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import *
from models.mlp_policy import DiagnormalPolicy, DiscretePolicy
from models.mlp_critic import ValueFunction
from models.mlp_net import Discriminator
from core.gail import GailUpdater
from core.agent import ActorCriticAgent
from core.trainer import ActorCriticTrainer
from core.evaluator import ActorCriticEvaluator


parser = argparse.ArgumentParser(description='PyTorch GAIL example')
parser.add_argument('--env-name', default="Hopper-v1", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--expert-traj-path', metavar='G',
                    help='path of the expert trajectories')
parser.add_argument('--render', action='store_true', default=False,
                    help='render the environment')
parser.add_argument('--log-std', type=float, default=0, metavar='G',
                    help='log std for the policy (default: 0)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.95, metavar='G',
                    help='gae (default: 0.95)')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--learning-rate', type=float, default=3e-4, metavar='G',
                    help='gae (default: 3e-4)')
parser.add_argument('--clip-epsilon', type=float, default=0.2, metavar='N',
                    help='clipping epsilon for PPO')
parser.add_argument('--num-threads', type=int, default=4, metavar='N',
                    help='number of threads for agent (default: 4)')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--min-batch-size', type=int, default=2048, metavar='N',
                    help='minimal batch size per PPO update (default: 2048)')
parser.add_argument('--max-iter-num', type=int, default=500, metavar='N',
                    help='maximal number of main iterations (default: 500)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--save-model-interval', type=int, default=0, metavar='N',
                    help="interval between saving model (default: 0, means don't save)")
args = parser.parse_args()


def env_factory(thread_id):
    env = gym.make(args.env_name)
    env.seed(args.seed + thread_id)
    return env


set_seed(args.seed)
torch.set_default_tensor_type('torch.DoubleTensor')
state_dim, action_dim, is_disc_action = get_gym_info(env_factory)
running_state = ZFilter((state_dim,), clip=5)

# Define actor, critic and discriminator
if is_disc_action:
    policy_net = DiscretePolicy(state_dim, action_dim)
    action_dim = 1
else:
    policy_net = DiagnormalPolicy(state_dim, action_dim, log_std=args.log_std)
value_net = ValueFunction(state_dim)
discrim_net = Discriminator(state_dim + action_dim)
nets = {'policy': policy_net, 'value': value_net, 'discrim': discrim_net}

if use_gpu and args.gpu:
    for name, net in nets.items():
        nets[name] = net.cuda()

# Define the optimizers of actor, critic and discriminator
optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=args.learning_rate)
optimizer_value = torch.optim.Adam(value_net.parameters(), lr=args.learning_rate)
optimizer_discrim = torch.optim.Adam(discrim_net.parameters(), lr=args.learning_rate)
optimizers = {}

# optimization epoch number and batch size for PPO
optim_epochs = 5
optim_batch_size = 4096

# Load trajectory
expert_traj, running_state = pickle.load(open(args.expert_traj_path, "rb"))


def expert_reward(state, action):
    state_action = np_to_tensor(np.hstack([state, action]))
    return -math.log(discrim_net(Variable(state_action, volatile=True)).data.numpy()[0])


cfg = Cfg(parse=args)
agent = ActorCriticAgent("Gail", env_factory, policy_net, value_net, cfg,
                         custom_reward=expert_reward, running_state=running_state)
agent.add_model('discrim', discrim_net)
gail = GailUpdater(nets, optimizers, cfg)
evaluator = ActorCriticEvaluator(agent, cfg)
trainer = ActorCriticTrainer(agent, gail, cfg, evaluator)
trainer.start()


# agent = ActorCriticAgent('GAIL', env_factory, policy_net, value_net, custom_reward=expert_reward,
#                          running_state=running_state, render=args.render, num_threads=args.num_threads)


def update_params(batch, i_iter):
    states = torch.from_numpy(np.stack(batch.state))
    actions = torch.from_numpy(np.stack(batch.action))
    rewards = torch.from_numpy(np.stack(batch.reward))
    masks = torch.from_numpy(np.stack(batch.mask).astype(np.float64))
    if use_gpu:
        states, actions, rewards, masks = states.cuda(), actions.cuda(), rewards.cuda(), masks.cuda()
    values = value_net(Variable(states, volatile=True)).data
    fixed_log_probs = policy_net.get_log_prob(Variable(states, volatile=True), Variable(actions)).data

    """get advantage estimation from the trajectories"""
    advantages, returns = estimate_advantages(rewards, masks, values, args.gamma, args.tau, use_gpu)

    lr_mult = max(1.0 - float(i_iter) / args.max_iter_num, 0)

    """update discriminator"""
    for _ in range(3):
        expert_state_actions = Tensor(expert_traj)
        if use_gpu:
            expert_state_actions = expert_state_actions.cuda()
        g_o = discrim_net(Variable(torch.cat([states, actions], 1)))
        e_o = discrim_net(Variable(expert_state_actions))
        optimizer_discrim.zero_grad()
        discrim_loss = discrim_criterion(g_o, Variable(ones((states.shape[0], 1)))) + \
            discrim_criterion(e_o, Variable(zeros((expert_traj.shape[0], 1))))
        discrim_loss.backward()
        optimizer_discrim.step()

    """perform mini-batch PPO update"""
    optim_iter_num = int(math.ceil(states.shape[0] / optim_batch_size))
    for _ in range(optim_epochs):
        perm = np.arange(states.shape[0])
        np.random.shuffle(perm)
        perm = LongTensor(perm)
        if use_gpu:
            perm = perm.cuda()
        states, actions, returns, advantages, fixed_log_probs = \
            states[perm], actions[perm], returns[perm], advantages[perm], fixed_log_probs[perm]

        for i in range(optim_iter_num):
            ind = slice(i * optim_batch_size, min((i + 1) * optim_batch_size, states.shape[0]))
            states_b, actions_b, advantages_b, returns_b, fixed_log_probs_b = \
                states[ind], actions[ind], advantages[ind], returns[ind], fixed_log_probs[ind]

            ppo_step(policy_net, value_net, optimizer_policy, optimizer_value, 1, states_b, actions_b, returns_b,
                     advantages_b, fixed_log_probs_b, lr_mult, args.learning_rate, args.clip_epsilon, args.l2_reg)


def main_loop():
    for i_iter in range(args.max_iter_num):
        """generate multiple trajectories that reach the minimum batch_size"""
        if use_gpu:
            discrim_net.cpu()
        batch, log = agent.collect_samples(args.min_batch_size)
        if use_gpu:
            discrim_net.cuda()

        t0 = time.time()
        update_params(batch, i_iter)
        t1 = time.time()

        if i_iter % args.log_interval == 0:
            print('{}\tT_sample {:.4f}\tT_update {:.4f}\texpert_R_avg {:.2f}\tR_avg {:.2f}'.format(
                i_iter, log['sample_time'], t1-t0, log['avg_c_reward'], log['avg_reward']))

        if args.save_model_interval > 0 and (i_iter+1) % args.save_model_interval == 0:
            if use_gpu:
                policy_net.cpu(), value_net.cpu(), discrim_net.cpu()
            pickle.dump((policy_net, value_net, discrim_net), open(os.path.join(asset_dir(), 'learned_models/{}_gail.p'.format(args.env_name)), 'wb'))
            if use_gpu:
                policy_net.cuda(), value_net.cuda(), discrim_net.cuda()


main_loop()
