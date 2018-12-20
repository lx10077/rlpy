import argparse
import gym
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils import *
from models.mlp_policy import DiagnormalPolicy, DiscretePolicy
from models.mlp_critic import ValueFunction, QFunction
from core.agent import ActorCriticAgent
from core.trainer import ActorCriticTrainer
from core.evaluator import ActorCriticEvaluator
from config.sac.sac_opt import SacUpdater

parser = argparse.ArgumentParser(description='PyTorch ERPPO example')
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
parser.add_argument('--l2-reg', type=float, default=0, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--learning-rate', type=float, default=3e-4, metavar='G',
                    help='learning rate (default: 3e-4)')
parser.add_argument('--optim-epochs', type=int, default=5, metavar='N',
                    help='number of updates in each timestep (default: 5)')
parser.add_argument('--optim-value-iternum', type=int, default=1, metavar='N',
                    help='number of value updates in each optim epoch (default: 1)')
parser.add_argument('--optim-qvalue-iternum', type=int, default=1, metavar='N',
                    help='number of value updates in each optim epoch (default: 1)')
parser.add_argument('--optim-batch-size', type=int, default=256, metavar='N',
                    help='optim batch size per PPO update (default: 256)')
parser.add_argument('--clip-epsilon', type=float, default=0.2, metavar='N',
                    help='clipping epsilon for PPO')
parser.add_argument('--num-threads', type=int, default=4, metavar='N',
                    help='number of threads for agent (default: 4)')
parser.add_argument('--seed', type=int, default=2, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--dis', type=str, default='')
parser.add_argument('--min-batch-size', type=int, default=2048, metavar='N',
                    help='minimal batch size per PPO update (default: 2048)')
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
assert not is_disc_action
policy_net = DiagnormalPolicy(state_dim, action_dim, log_std=args.log_std)
value_net = ValueFunction(state_dim)
qvalue_net = QFunction(state_dim + action_dim)

device = torch.device("cuda" if use_gpu and args.gpu else "cpu")
if use_gpu and args.gpu:
    policy_net = policy_net.to(device)
    value_net = value_net.to(device)
    qvalue_net = qvalue_net.to(device)

optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=args.learning_rate)
optimizer_value = torch.optim.Adam(value_net.parameters(), lr=args.learning_rate)
optimizer_qvalue = torch.optim.Adam(qvalue_net.parameters(), lr=args.learning_rate)

cfg = Cfg(parse=args)
agent = ActorCriticAgent("Sac" + args.dis, env_factory, policy_net, value_net, cfg,
                         running_state=running_state)
sac = SacUpdater(policy_net, value_net, qvalue_net,
                 optimizer_policy, optimizer_value, optimizer_qvalue, cfg)
evaluator = ActorCriticEvaluator(agent, cfg)
trainer = ActorCriticTrainer(agent, sac, cfg, evaluator)
trainer.start()
