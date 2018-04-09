from torch.autograd import Variable
from utils.tools import *


class ActorCriticTester(object):
    def __init__(self, agent, cfg):
        self.agent = agent
        self.cfg = cfg
        self.asset_dir = assets_dir()
        self.eval_dir = set_dir(self.asset_dir, "evaluations")

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

    def start(self, num_epsd=None):
        if num_epsd is None:
            num_epsd = self.num_epsd
        self._check()
        self._eval(num_epsd)
        self.save_evals()

    def eval(self, iter_i):
        avg_rewards, eval_rewards = self._eval(self.num_epsd)
        self.eval_iters.append(iter_i)
        self.eval_rewards.append(eval_rewards)
        title = "Test Rewards of " + self.id
        population_plot(self.eval_iters, self.eval_rewards, title, self.eval_dir)

    def save_evals(self):
        print("[Save] Saving evaluation results...")
        file = self.eval_dir + "/" + self.id
        self.eval_iters = np.array(self.eval_iters)
        self.eval_rewards = np.array(self.eval_rewards)
        np.savez(file, eval_iters=self.eval_iters, eval_rewards=self.eval_rewards)

    def _eval(self, num_epsd):
        eval_env = self.env_factory(3333)
        total_rewards = np.zeros(num_epsd)
        epsd_idx = 0
        epsd_iters = 0
        state = eval_env.reset()
        while epsd_idx < num_epsd:
            if self.agent.running_state is not None:
                state = self.agent.running_state(state, update=False)

            state_var = Variable(self.agent.tensor(state).unsqueeze(0), volatile=True)
            action = self.policy.select_action(state_var)
            next_state, reward, done, _ = eval_env.step(action)
            total_rewards[epsd_idx] += reward
            epsd_iters += 1
            state = next_state

            if done or epsd_iters >= self.max_epsd_iters:
                print('>>> Eval: [%2d/%d], rewards: %s' % (epsd_idx + 1, num_epsd, total_rewards[epsd_idx]))

                if epsd_idx < num_epsd - 1:  # leave last reset to next run
                    state = eval_env.reset()

                epsd_idx += 1
                epsd_iters = 0

        avg_rewards = total_rewards.mean()
        print('>>> Eval: avg total rewards: %s' % avg_rewards)
        eval_env.close()
        del eval_env
        return avg_rewards, total_rewards

    def _check(self):
        if self.agent.running_state is None:
            print("[Warning] No running states.")
