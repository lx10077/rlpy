import argparse
import gym
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import *
from models.mlp_policy import DiagnormalPolicy, Policy, DiscretePolicy
from models.mlp_critic import Value, ValueFunction
from torch.autograd import Variable
from core.ppo import ppo_step, ClipPpoUpdater
from core.common import estimate_advantages
from core.agent import ActorCriticAgent
from core.trainer import ActorCriticTrainer
from core.evaluator import ActorCriticTester

parser = argparse.ArgumentParser(description='PyTorch PPO example')
parser.add_argument('--env-name', default="HalfCheetah-v2", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--model-path', metavar='G',
                    help='path of pre-trained model')
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
                    help='learning rate (default: 3e-4)')
parser.add_argument('--optim-epochs', type=int, default=10, metavar='N',
                    help='number of updates in each timestep (default: 5)')
parser.add_argument('--optim-value-iternum', type=int, default=1, metavar='N',
                    help='number of value updates in each optim epoch (default: 1)')
parser.add_argument('--optim-batch-size', type=int, default=32, metavar='N',
                    help='optim batch size per PPO update (default: 32)')
parser.add_argument('--clip-epsilon', type=float, default=0.2, metavar='N',
                    help='clipping epsilon for PPO')
parser.add_argument('--num-threads', type=int, default=4, metavar='N',
                    help='number of threads for agent (default: 4)')
parser.add_argument('--seed', type=int, default=2, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--min-batch-size', type=int, default=2048, metavar='N',
                    help='minimal batch size per PPO update (default: 2048)')
parser.add_argument('--max-iter-num', type=int, default=2000, metavar='N',
                    help='maximal number of main iterations (default: 500)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--save-model-interval', type=int, default=0, metavar='N',
                    help="interval between saving model (default: 0, means don't save)")
parser.add_argument('--eval-model-interval', type=int, default=0, metavar='N',
                    help="interval between saving model (default: 0, means don't save)")
args = parser.parse_args()
torch.set_default_tensor_type('torch.DoubleTensor')
set_seed(args.seed)


def env_factory(thread_id):
    env = gym.make(args.env_name)
    env.seed(args.seed + thread_id)
    return env


state_dim, action_dim, is_disc_action = get_gym_info(env_factory)
running_state = ZFilter((state_dim,), clip=5)

# Define actor, critic and their optimizers
if is_disc_action:
    policy_net = DiscretePolicy(state_dim, action_dim)
else:
    policy_net = DiagnormalPolicy(state_dim, action_dim, log_std=args.log_std)
value_net = ValueFunction(state_dim)

if use_gpu:
    policy_net = policy_net.cuda()
    value_net = value_net.cuda()

optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=args.learning_rate)
optimizer_value = torch.optim.Adam(value_net.parameters(), lr=args.learning_rate)

# optimization epoch number and batch size for PPO
optim_epochs = 5
optim_batch_size = 4096

cfg = Cfg(parse=args)
agent = ActorCriticAgent("ClipPPO", env_factory, policy_net, value_net, cfg, running_state=running_state)
clip_ppo = ClipPpoUpdater(policy_net, value_net, optimizer_policy, optimizer_value, cfg)
evaluator = ActorCriticTester(agent, cfg)
trainer = ActorCriticTrainer(agent, clip_ppo, cfg, evaluator)
trainer.start()


def update_params(batch, i_iter):
    states = torch.from_numpy(np.stack(batch.state))
    actions = torch.from_numpy(np.stack(batch.action))
    rewards = torch.from_numpy(np.stack(batch.reward))
    masks = torch.from_numpy(np.stack(batch.mask).astype(np.float64))
    if use_gpu:
        states, actions, rewards, masks = states.cuda(), actions.cuda(), rewards.cuda(), masks.cuda()
    values = value_net(Variable(states, volatile=True)).data

    """get advantage estimation from the trajectories"""
    advantages, returns = estimate_advantages(rewards, masks, values, args.gamma, args.tau, use_gpu)

    fixed_log_probs = policy_net.get_log_prob(Variable(states, volatile=True), Variable(actions)).data

    lr_mult = max(1.0 - float(i_iter) / args.max_iter_num, 0)

    """perform mini-batch PPO update"""
    optim_iter_num = int(math.ceil(states.shape[0] / optim_batch_size))
    for _ in range(optim_epochs):
        perm = np.arange(states.shape[0])
        np.random.shuffle(perm)
        perm = LongTensor(perm).cuda() if use_gpu else LongTensor(perm)

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
        batch, log = agent.collect_samples(args.min_batch_size)
        t0 = time.time()
        update_params(batch, i_iter)
        t1 = time.time()

        if i_iter % args.log_interval == 0:
            print('{}\tT_sample {:.4f}\tT_update {:.4f}\tR_min {:.2f}\tR_max {:.2f}\tR_avg {:.2f}'.format(
                i_iter, log['sample_time'], t1-t0, log['min_reward'], log['max_reward'], log['avg_reward']))

        if args.save_model_interval > 0 and (i_iter+1) % args.save_model_interval == 0:
            if use_gpu:
                policy_net.cpu(), value_net.cpu()
            pickle.dump((policy_net, value_net, running_state),
                        open(os.path.join(assets_dir(), 'learned_models/{}_ppo.p'.format(args.env_name)), 'wb'))
            if use_gpu:
                policy_net.cuda(), value_net.cuda()


# main_loop()
