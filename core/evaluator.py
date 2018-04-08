from torch.autograd import Variable
from utils.tools import *
from tensorboardX import SummaryWriter


class ActorCriticTester(object):
    def __init__(self, agent, cfg):
        self.agent = agent
        self.cfg = cfg
        self.asset_dir = assets_dir()

        # get some components from the agent
        self.env_factory = self.agent.env_factory
        self.tensor = self.agent.tensor
        self.policy = self.agent.policy
        self.id = self.agent.id

        # parameters for evaluation
        self.num_epsd = cfg["num_epsd"] if "num_epsd" in cfg else 10
        self.max_epsd_iters = cfg["max_epsd_iters"] if "max_epsd_iters" in cfg else 10000
        self.eval_iters = []
        self.eval_rewards = [np.zeros(self.num_epsd)]  # elements are numpy array
        self.best_avg_rewards = -np.inf

        # parameters for records
        self.record_iters = []
        self.record_rewards = []
        self.record_custom_rewards = []
        self.writer = SummaryWriter(self.asset_dir + "/" + self.id)

    def record(self, iter_i, timestep_log):
        self.record_iters.append(iter_i)
        self.record_rewards.append(timestep_log["avg_reward"])
        if "avg_c_reward" in timestep_log:
            self.record_custom_rewards.append(timestep_log["avg_c_reward"])

    def save_records(self):
        file = self.asset_dir + "records/" + self.id
        self.record_rewards = np.array(self.record_rewards)
        if len(self.record_custom_rewards) > 0:
            self.record_custom_rewards = np.array(self.record_custom_rewards)
            np.savez(file, rewards=self.record_custom_rewards,
                     custom_rewards=self.record_custom_rewards)
        else:
            np.savez(file, rewards=self.record_rewards)

    def monitor(self, iter_i, timestep_log):
        log_plot(self.writer, timestep_log, iter_i)

    def eval(self, iter_i):
        avg_rewards, eval_rewards = self._eval()
        self.eval_iters.append(iter_i)
        self.eval_rewards.append(eval_rewards)
        title = "Test Rewards of " + self.id
        population_plot(self.eval_iters, self.eval_rewards, title, self.asset_dir)

    def _eval(self):
        eval_env = self.env_factory(3333)
        total_rewards = np.zeros(self.num_epsd)
        epsd_idx = 0
        epsd_iters = 0
        state = eval_env.reset()
        while epsd_idx < self.num_epsd:
            if self.agent.running_state is not None:
                state = self.agent.running_state(state, update=False)

            state_var = Variable(self.agent.tensor(state).unsqueeze(0), volatile=True)
            action = self.policy.select_action(state_var)
            next_state, reward, done, _ = eval_env.step(action)
            total_rewards[epsd_idx] += reward
            epsd_iters += 1
            state = next_state

            if done or epsd_iters >= self.max_epsd_iters:
                print('>>> Eval: [%2d/%d], rewards: %s' % (epsd_idx + 1, self.num_epsd, total_rewards[epsd_idx]))

                if epsd_idx < self.num_epsd - 1:  # leave last reset to next run
                    state = eval_env.reset()

                epsd_idx += 1
                epsd_iters = 0

        avg_rewards = total_rewards.mean()
        print('>>> Eval: avg total rewards: %s' % avg_rewards)
        eval_env.close()
        del eval_env
        return avg_rewards, total_rewards
