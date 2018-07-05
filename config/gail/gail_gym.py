import argparse
import gym
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils import *
from models.mlp_policy import DiagnormalPolicy, DiscretePolicy
from models.mlp_critic import ValueFunction
from models.mlp_net import Discriminator
from config.gail.gail_opt import GailUpdater
from config.gail.traj_giver import TrajGiver
from core.agent import ActorCriticAgent
from core.trainer import ActorCriticTrainer
from core.evaluator import ActorCriticEvaluator


parser = argparse.ArgumentParser(description='PyTorch GAIL example')
parser.add_argument('--env-name', default="Hopper-v2", metavar='G',
                    help='name of the environment to run')
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
parser.add_argument('--max-kl', type=float, default=1e-2, metavar='G',
                    help='max kl value (default: 1e-2)')
parser.add_argument('--damping', type=float, default=1e-2, metavar='G',
                    help='damping (default: 1e-2)')
parser.add_argument('--threshold', type=int, default=0, metavar='T',
                    help='threshold to be an expert trajectory (default: 0, means choose the first one)')
parser.add_argument('--algo', type=str, default='clipppo', metavar='T', choices=['clipppo', 'trpo'],
                    help="which algo to optimize params from (default: clipppo)")
parser.add_argument('--optim-discrim-iternum', type=int, default=3, metavar='N',
                    help='number of discrim updates in each timestep (default: 3)')
parser.add_argument('--optim-epochs', type=int, default=5, metavar='N',
                    help='number of updates in each timestep (default: 5)')
parser.add_argument('--optim-value-iternum', type=int, default=1, metavar='N',
                    help='number of value updates in each optim epoch (default: 1)')
parser.add_argument('--optim-batch-size', type=int, default=256, metavar='N',
                    help='optim batch size per PPO update (default: 256)')
parser.add_argument('--learning-rate', type=float, default=3e-4, metavar='G',
                    help='gae (default: 3e-4)')
parser.add_argument('--clip-epsilon', type=float, default=0.2, metavar='N',
                    help='clipping epsilon for PPO')
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
state_dim, action_dim, is_disc_action = get_gym_info(env_factory)
# running_state = ZFilter((state_dim,), clip=5)
running_state = None

# Define actor, critic and discriminator
if is_disc_action:
    policy_net = DiscretePolicy(state_dim, action_dim)
    action_dim = 1
else:
    policy_net = DiagnormalPolicy(state_dim, action_dim, log_std=args.log_std)
value_net = ValueFunction(state_dim)
discrim_net = Discriminator(state_dim + action_dim)
nets = {'policy': policy_net,
        'value': value_net,
        'discrim': discrim_net}

if use_gpu and args.gpu:
    for name, net in nets.items():
        nets[name] = net.cuda()

# Define the optimizers of actor, critic and discriminator
optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=args.learning_rate)
optimizer_value = torch.optim.Adam(value_net.parameters(), lr=args.learning_rate)
optimizer_discrim = torch.optim.Adam(discrim_net.parameters(), lr=args.learning_rate)
optimizers = {'optimizer_policy': optimizer_policy,
              'optimizer_value': optimizer_value,
              'optimizer_discrim': optimizer_discrim}


def expert_reward(state, action):
    with torch.no_grad():
        state_action = np_to_tensor(np.hstack([state, action]))
        custom_reward = -math.log(discrim_net(state_action).item())
    return custom_reward


cfg = Cfg(parse=args)
agent = ActorCriticAgent("Gail" + args.dis, env_factory, policy_net, value_net, cfg,
                         custom_reward=expert_reward, running_state=running_state)
agent.add_model('discrim', discrim_net)

# Load or make expert trajectory
expert_dl = TrajGiver(cfg)(prefer=args.algo, threshold=args.threshold)

gail = GailUpdater(nets, optimizers, cfg)
gail.load_traj(expert_dl)

evaluator = ActorCriticEvaluator(agent, cfg)
trainer = ActorCriticTrainer(agent, gail, cfg, evaluator)
trainer.start()
