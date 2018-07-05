import argparse
import gym
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils import *
from models.mlp_policy import DiagnormalPolicy, DiscretePolicy
from models.mlp_critic import ValueFunction
from core.a2c import A2cUpdater
from core.agent import ActorCriticAgent
from core.trainer import ActorCriticTrainer
from core.evaluator import ActorCriticEvaluator


parser = argparse.ArgumentParser(description='PyTorch A2C example')
parser.add_argument('--env-name', default="Hopper-v2", metavar='G',
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
parser.add_argument('--gpu', action='store_true', default=False,
                    help='use gpu(default: False)')
parser.add_argument('--learning-rate', type=float, default=3e-4, metavar='G',
                    help='learning rate (default: 7e-4)')
parser.add_argument('-lr-policy', type=float, default=9e-4, metavar='G',
                    help='learning rate (default: 9e-4)')
parser.add_argument('--lr-value', type=float, default=1e-3, metavar='G',
                    help='learning rate (default: 1e-3)')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--num-threads', type=int, default=4, metavar='N',
                    help='number of threads for agent (default: 4)')
parser.add_argument('--seed', type=int, default=2, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--dis', type=str, default='')
parser.add_argument('--min-batch-size', type=int, default=2048, metavar='N',
                    help='minimal batch size per A2C update (default: 2048)')
parser.add_argument('--max-iter-num', type=int, default=500, metavar='N',
                    help='maximal number of main iterations (default: 500)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 1)')
parser.add_argument('--save-model-interval', type=int, default=100, metavar='N',
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

running_state = ZFilter((state_dim,), clip=5)

optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=args.lr_policy)
optimizer_value = torch.optim.Adam(value_net.parameters(), lr=args.lr_value)

cfg = Cfg(parse=args)
agent = ActorCriticAgent("A2c" + args.dis, env_factory, policy_net, value_net, cfg, running_state=running_state)
a2c = A2cUpdater(policy_net, value_net, optimizer_policy, optimizer_value, cfg)
evaluator = ActorCriticEvaluator(agent, cfg)
trainer = ActorCriticTrainer(agent, a2c, cfg, evaluator)
trainer.start()
