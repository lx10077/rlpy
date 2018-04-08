import argparse
import gym
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import *
from models.mlp_policy import DiagnormalPolicy, Policy, DiscretePolicy
from models.mlp_critic import Value, ValueFunction
from torch.autograd import Variable
from core.trpo import TrpoUpdater
from core.common import estimate_advantages
from core.agent import ActorCriticAgent
from core.trainer import ActorCriticTrainer
from core.evaluator import ActorCriticTester


parser = argparse.ArgumentParser(description='PyTorch TRPO example')
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
parser.add_argument('--max-kl', type=float, default=1e-2, metavar='G',
                    help='max kl value (default: 1e-2)')
parser.add_argument('--damping', type=float, default=1e-2, metavar='G',
                    help='damping (default: 1e-2)')
parser.add_argument('--num-threads', type=int, default=4, metavar='N',
                    help='number of threads for agent (default: 4)')
parser.add_argument('--seed', type=int, default=2, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--min-batch-size', type=int, default=2048, metavar='N',
                    help='minimal batch size per TRPO update (default: 2048)')
parser.add_argument('--max-iter-num', type=int, default=650, metavar='N',
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

if use_gpu and args.gpu:
    policy_net = policy_net.cuda()
    value_net = value_net.cuda()


cfg = Cfg(parse=args)
agent = ActorCriticAgent("Trpo", env_factory, policy_net, value_net, cfg, running_state=running_state)
trpo = TrpoUpdater(policy_net, value_net, args.max_kl, args.damping, args.l2_reg, nsteps=10, use_fim=False)
evaluator = ActorCriticTester(agent, cfg)
trainer = ActorCriticTrainer(agent, trpo, cfg, evaluator)
trainer.start()

def update_params(batch):
    states = torch.from_numpy(np.stack(batch.state))
    actions = torch.from_numpy(np.stack(batch.action))
    rewards = torch.from_numpy(np.stack(batch.reward))
    masks = torch.from_numpy(np.stack(batch.mask).astype(np.float64))
    if use_gpu:
        states, actions, rewards, masks = states.cuda(), actions.cuda(), rewards.cuda(), masks.cuda()
    values = value_net(Variable(states, volatile=True)).data

    """get advantage estimation from the trajectories"""
    advantages, returns = estimate_advantages(rewards, masks, values, args.gamma, args.tau, use_gpu)

    batchs = {}
    batchs["states"] = states
    batchs["actions"] = actions
    batchs["advantages"] = advantages
    batchs["value_targets"] = returns

    """perform TRPO update"""
    # trpo_step(policy_net, value_net, states, actions, returns, advantages, args.max_kl, args.damping, args.l2_reg)
    log = {}
    log = trpo(batchs, log)


def main_loop():
    for i_iter in range(args.max_iter_num):
        """generate multiple trajectories that reach the minimum batch_size"""
        batch, log = agent.collect_samples(args.min_batch_size)
        t0 = time.time()
        update_params(batch)
        t1 = time.time()

        if i_iter % args.log_interval == 0:
            print('{}\tT_sample {:.4f}\tT_update {:.4f}\tR_min {:.2f}\tR_max {:.2f}\tR_avg {:.2f}'.format(
                i_iter, log['sample_time'], t1-t0, log['min_reward'], log['max_reward'], log['avg_reward']))

        if args.save_model_interval > 0 and (i_iter+1) % args.save_model_interval == 0:
            if use_gpu:
                policy_net.cpu(), value_net.cpu()
            pickle.dump((policy_net, value_net, running_state),
                        open(os.path.join(assets_dir(), 'learned_models/{}_trpo.p'.format(args.env_name)), 'wb'))
            if use_gpu:
                policy_net.cuda(), value_net.cuda()


# main_loop()
