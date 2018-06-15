from utils.tools import trainlog_dir, asset_dir, set_dir
from tensorboardX import SummaryWriter
from plotly.graph_objs import Scatter, Line
import os
import plotly
import logging


trainlog_dir = trainlog_dir()
asset_dir = asset_dir()


def loggerconfig(log_file, verbose=2):
    logger = logging.getLogger()
    formatter = logging.Formatter('[%(levelname)-8s] (%(processName)-11s) %(message)s')
    file_handler = logging.FileHandler(log_file, 'w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    if verbose >= 2:
        logger.setLevel(logging.DEBUG)
    elif verbose >= 1:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)
    return logger


class Dlogger(object):
    def __init__(self, output_name):
        self.log_file = open(output_name, 'w')
        self.infos = {}

    def append(self, key, val):
        vals = self.infos.setdefault(key, [])
        vals.append(val)

    def log(self, extra_msg=''):
        msgs = [extra_msg]
        for key, vals in self.infos.items():
            msgs.append('%s %.6f' % (key, sum(vals)/len(vals)))
        msg = '\n'.join(msgs)
        self.log_file.write(msg + '\n')
        self.log_file.flush()
        self.infos = {}
        return msg

    def write(self, msg, mute=False):
        self.log_file.write(msg + '\n')
        self.log_file.flush()
        if not mute:
            print(msg)


# ====================================================================================== #
# Task and logger
# ====================================================================================== #
class task(object):
    def __init__(self, agent_id, cfg):
        self.name = agent_id
        self.cfg = cfg
        self.task_save_dir = os.path.join(trainlog_dir, self.name)
        self.set_subfiles()

    def set_subfiles(self):
        self.cfg.save_config(os.path.join(self.task_save_dir, 'cfg.json'))
        set_dir(self.task_save_dir, 'models')
        set_dir(self.task_save_dir, 'train.events')
        set_dir(self.task_save_dir, 'result_summary')

    @property
    def model_dir(self):
        return os.path.join(self.task_save_dir, 'models')

    @property
    def event_dir(self):
        return os.path.join(self.task_save_dir, 'train.events')

    @property
    def summary_dir(self):
        return os.path.join(self.task_save_dir, 'result_summary')


class logger(object):
    def __init__(self, agent, cfg):
        self.task = task(agent.id, cfg)
        self.writer = SummaryWriter(self.task.event_dir)
        self.train_log = loggerconfig(os.path.join(self.task.task_save_dir, 'trainlog.txt'))
        self.test_log = loggerconfig(os.path.join(self.task.task_save_dir, 'testlog.txt'))

    def record(self, i_iter, update_log):
        """Monitor training process by TensorboardX.

        Parameters
        ----------
        writer: SummaryWriter in TensorboardX, run "tensorboard --logdir runs".
        update_log: dict like {tag: value}, containing all sampling and updating information.
        i_iter: int, global step.
        """
        reward_dict = {}
        action_dict = {}
        if not update_log:
            self.write("Empty updating log!")
            return

        for tag, value in update_log.items():
            if "reward" in tag:
                reward_dict[tag] = value
            elif "action" in tag:
                action_dict[tag] = value
            else:
                self.writer.add_scalar(tag, value, i_iter)

        if 'total_reward' in reward_dict.keys():
            del reward_dict['total_reward']
        self.writer.add_scalars('reward', reward_dict, i_iter)
        raise NotImplementedError

    def write(self, msg, type='train'):
        raise NotImplementedError



# ====================================================================================== #
# Plotting and monitoring
# ====================================================================================== #
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



