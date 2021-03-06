import argparse
import gym
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils import *
from models.mlp_policy import DiagnormalPolicy
from models.mlp_critic import ValueFunction
from core.agent import ActorCriticAgent
from core.trainer import ActorCriticTrainer
from core.evaluator import ActorCriticEvaluator
from config.gbpo.gbpo_policy import AdditiveDiagnormalPolicy
from config.gbpo.gbpo_opt import GbpoUpdater


parser = argparse.ArgumentParser(description='PyTorch Gradient Boosting Policy Optimization example')
parser.add_argument('--env-name', default="Hopper-v2", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--model-path', metavar='G',
                    help='path of pre-trained model')
parser.add_argument('--policy-num', type=int, default=5, metavar='N',
                    help='number of policy (default: 5)')
parser.add_argument('--alpha', type=float, default=0.5, metavar='N',
                    help='interpolated coefficient for CPI')
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
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--learning-rate', type=float, default=3e-4, metavar='G',
                    help='learning rate (default: 3e-4)')
parser.add_argument('--optim-epochs', type=int, default=5, metavar='N',
                    help='number of updates in each timestep (default: 5)')
parser.add_argument('--optim-value-iternum', type=int, default=1, metavar='N',
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
parser.add_argument('--max-iter-num', type=int, default=1000, metavar='N',
                    help='maximal number of main iterations (default: 500)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 1)')
parser.add_argument('--save-model-interval', type=int, default=100, metavar='N',
                    help="interval between saving model (default: 0, means don't save)")
parser.add_argument('--eval-model-interval', type=int, default=10, metavar='N',
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
policy_net = AdditiveDiagnormalPolicy(state_dim, action_dim, args.policy_num, args.alpha, args.gpu)
temporary_policy = DiagnormalPolicy(state_dim, action_dim)
value_net = ValueFunction(state_dim)

device = torch.device("cuda" if use_gpu and args.gpu else "cpu")
if use_gpu and args.gpu:
    policy_net = policy_net.to(device)
    temporary_policy = temporary_policy.to(device)
    value_net = value_net.to(device)

optimizer_policy = torch.optim.Adam(temporary_policy.parameters(), lr=args.learning_rate)
optimizer_value = torch.optim.Adam(value_net.parameters(), lr=args.learning_rate)

cfg = Cfg(parse=args)
agent = ActorCriticAgent("GbPPO" + args.dis, env_factory, temporary_policy,
                         value_net, cfg, running_state=running_state)
gbpo_ppo = GbpoUpdater(policy_net, temporary_policy, value_net, optimizer_policy, optimizer_value, cfg)
evaluator = ActorCriticEvaluator(agent, cfg, policy=policy_net)
trainer = ActorCriticTrainer(agent, gbpo_ppo, cfg, evaluator)
trainer.start()
