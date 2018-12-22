import argparse
import gym
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils import *
from models.mlp_policy import DiagnormalPolicy, DiscretePolicy
from models.mlp_critic import ValueFunction, LinearVariate, QuadraticVariate, MlpVariate
from config.adcv.v_opt import VariateUpdater
from core.agent import ActorCriticAgent
from core.trainer import ActorCriticTrainer
from core.evaluator import ActorCriticEvaluator


parser = argparse.ArgumentParser(description='PyTorch ADCV example')
parser.add_argument('--env-name', default="Hopper-v2", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--variate', type=str, choices=['linear', 'mlp', 'quadratic'], metavar='V',
                    help='variate (default: mlp)')
parser.add_argument('--opt', type=str, choices=['minvar', 'fitq'], metavar='V', default='fitq',
                    help='variate optimization method (default: fitq)')
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
                    help='lr for variate (default: 3e-4)')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--max-kl', type=float, default=1e-2, metavar='G',
                    help='max kl value (default: 1e-2)')
parser.add_argument('--damping', type=float, default=1e-2, metavar='G',
                    help='damping (default: 1e-2)')
parser.add_argument('--optim-adcv-iternum', type=int, default=3, metavar='N',
                    help='number of varuate updates in each timestep (default: 3)')
parser.add_argument('--optim-epochs', type=int, default=5, metavar='N',
                    help='number of updates in each timestep (default: 5)')
parser.add_argument('--opt-iterval', type=int, default=50, metavar='N',
                    help='variate optim inerval (default: 50)')
parser.add_argument('--optim-value-iternum', type=int, default=1, metavar='N',
                    help='number of value updates in each optim epoch (default: 1)')
parser.add_argument('--num-threads', type=int, default=4, metavar='N',
                    help='number of threads for agent (default: 4)')
parser.add_argument('--seed', type=int, default=1, metavar='N',
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


def env_factory(thread_id):
    env = gym.make(args.env_name)
    env.seed(args.seed + thread_id)
    return env


set_seed(args.seed)
torch.set_default_tensor_type('torch.DoubleTensor')
state_dim, action_dim, _ = get_gym_info(env_factory)
running_state = ZFilter((state_dim,), clip=5)

# Define actor, critic and discriminator
hidden = [10 * state_dim, math.ceil(math.sqrt(50 * state_dim)), 5]
policy_net = DiagnormalPolicy(state_dim, action_dim, hidden=hidden, log_std=args.log_std)
value_net = ValueFunction(state_dim)
del hidden

if args.variate == 'linear':
    variate_net = LinearVariate(state_dim, action_dim)
elif args.variate == 'quadratic':
    variate_net = QuadraticVariate(state_dim, action_dim)
else:
    variate_net = MlpVariate(state_dim, action_dim)

nets = {'policy': policy_net,
        'value': value_net,
        'variate': variate_net}

if use_gpu and args.gpu:
    for name, net in nets.items():
        nets[name] = net.cuda()

# Define the optimizers of actor, critic and discriminator
optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=9e-4/math.sqrt(5 * state_dim))
optimizer_value = torch.optim.Adam(value_net.parameters(), lr=1e-4/math.sqrt(5 * state_dim))
optimizer_variate = torch.optim.Adam(variate_net.parameters(), lr=args.learning_rate)
optimizers = {'optimizer_policy': optimizer_policy,
              'optimizer_value': optimizer_value,
              'optimizer_variate': optimizer_variate}

cfg = Cfg(parse=args)
exp_name = "Adcv-" + args.variate + "-" + args.opt + args.dis
agent = ActorCriticAgent(exp_name, env_factory, policy_net, value_net, cfg,
                         running_state=running_state)
adcv = VariateUpdater(nets, optimizers, cfg)
evaluator = ActorCriticEvaluator(agent, cfg)
trainer = ActorCriticTrainer(agent, adcv, cfg, evaluator)
trainer.start()
