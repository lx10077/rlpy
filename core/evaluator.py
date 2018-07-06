from plotly.graph_objs import Scatter, Line
from utils.torchs import *
import numpy as np
import plotly
import os


class ActorCriticEvaluator(object):
    def __init__(self, agent, cfg):
        self.agent = agent
        self.cfg = cfg
        self.log = None
        self.summary_dir = None

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
        self.eval_env = self.env_factory(3333)
        self.gpu = cfg["gpu"] if "gpu" in cfg else False

    def set_logger(self, log):
        self.log = log
        if not self.log.ready_for_test:
            self.log.prepare_for_test()
        self.summary_dir = self.log.task.summary_dir

    def place_models_on_cpu(self):
        self.agent.place_models_on_cpu()

    def place_models_on_gpu(self):
        self.agent.place_models_on_gpu()

    def eval(self, iter_i, viz=False):
        assert self.log is not None
        if use_gpu and self.gpu:
            self.place_models_on_cpu()

        avg_rewards, eval_rewards = self._eval(self.num_epsd)

        if use_gpu and self.gpu:
            self.place_models_on_gpu()

        self.eval_iters.append(iter_i)
        self.eval_rewards.append(eval_rewards)
        title = "Test Rewards of " + self.id
        if viz:
            population_plot(self.eval_iters, self.eval_rewards, title, self.summary_dir)
        return {'ravg': avg_rewards, 'rs': eval_rewards}

    def _eval(self, num_epsd):
        total_rewards = np.zeros(num_epsd)
        epsd_idx = 0
        epsd_iters = 0
        state = self.eval_env.reset()
        while epsd_idx < num_epsd:
            if self.agent.running_state is not None:
                state = self.agent.running_state(state, update=False)

            state_var = self.agent.tensor(state).unsqueeze(0)
            action = self.policy.select_action(state_var)
            next_state, reward, done, _ = self.eval_env.step(action)
            total_rewards[epsd_idx] += reward
            epsd_iters += 1
            state = next_state

            if done or epsd_iters >= self.max_epsd_iters:
                # print('>>> Eval: [%2d/%d], rewards: %s' % (epsd_idx + 1, num_epsd, total_rewards[epsd_idx]))

                if epsd_idx < num_epsd - 1:  # leave last reset to next run
                    state = self.eval_env.reset()

                epsd_idx += 1
                epsd_iters = 0

        avg_rewards = total_rewards.mean()
        # print('>>> Eval: avg total rewards: %s' % avg_rewards)
        return avg_rewards, total_rewards


def population_plot(xs, ys, title, path):
    """Plots min, max and mean + standard deviation bars of a population over time.

    Parameters
    ----------
    xs: iterations, list or numpy array, shape (N, )
    ys: sum of rewards, list or numpy array, shape (N, num_epsd)
    title: figure title
    path: saving dir
    """
    max_colour, mean_colour, std_colour = 'rgb(0, 132, 180)', 'rgb(0, 172, 237)', 'rgba(29, 202, 255, 0.2)'

    xs, ys = np.array(xs), np.array(ys)
    ys_min, ys_max = ys.min(1).squeeze(), ys.max(1).squeeze()
    ys_mean, ys_std = ys.mean(1).squeeze(), ys.std(1).squeeze()
    ys_upper, ys_lower = ys_mean + ys_std, ys_mean - ys_std

    trace_max = Scatter(x=xs, y=ys_max, line=Line(color=max_colour, dash='dash'), name='Max')
    trace_upper = Scatter(x=xs, y=ys_upper, line=Line(color='transparent'),
                          name='+1 Std. Dev.', showlegend=False)
    trace_mean = Scatter(x=xs, y=ys_mean, fill='tonexty',
                         fillcolor=std_colour, line=Line(color=mean_colour), name='Mean')
    trace_lower = Scatter(x=xs, y=ys_lower, fill='tonexty',
                          fillcolor=std_colour, line=Line(color='transparent'),
                          name='-1 Std. Dev.', showlegend=False)
    trace_min = Scatter(x=xs, y=ys_min, line=Line(color=max_colour, dash='dash'), name='Min')

    plotly.offline.plot({
        'data': [trace_upper, trace_mean, trace_lower, trace_min, trace_max],
        'layout': dict(title=title, xaxis={'title': 'Iteration'}, yaxis={'title': title})
    }, filename=os.path.join(path, title + '.html'), auto_open=False)
