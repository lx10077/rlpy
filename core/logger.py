from utils.tools import trainlog_dir, set_dir
from tensorboardX import SummaryWriter
import os
import logging


trainlog_dir = trainlog_dir()
config_dir = set_dir(trainlog_dir, 'config')


# ====================================================================================== #
# Task and logger
# ====================================================================================== #
class Task(object):
    def __init__(self, agent_id, cfg):
        self.name = agent_id
        self.cfg = cfg
        self.task_save_dir = set_dir(config_dir, self.name)
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


def loggerconfig(log_file, verbose=2, name='root'):
    """
    Verbose: critical=6 > error=5 > warning=4 > info=3 > debug=2 > notset=1
    """
    def rtlevel(verb):
        if verb >= 2:
            return logging.DEBUG
        elif verb == 1:
            return logging.INFO
        else:
            return logging.WARNING
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    filehdlr = logging.FileHandler(log_file)
    console = logging.StreamHandler()
    filehdlr.setLevel(logging.DEBUG)
    console.setLevel(rtlevel(verbose))
    formatter = logging.Formatter(fmt='[%(asctime)s] %(levelname)-7s: %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    filehdlr.setFormatter(formatter)
    console.setFormatter(formatter)
    logger.addHandler(filehdlr)
    logger.addHandler(console)
    return logger


class Logger(object):
    def __init__(self, log_id, cfg, verbose=2):
        self.task = Task(log_id, cfg)
        self.writer = SummaryWriter(self.task.event_dir)
        self.train_log = loggerconfig(os.path.join(self.task.task_save_dir, 'trainlog.txt'), verbose, 'trainlog')
        self.test_log = loggerconfig(os.path.join(self.task.task_save_dir, 'testlog.txt'), verbose+1, 'testlog')

    def record(self, i_iter, update_log, show_r_range=False):
        """SummaryWriter in TensorboardX, run 'tensorboard --logdir runs'.
        """
        reward_dict = {}
        if not update_log:
            self.write("Empty updating log!", 4)
            return

        for tag, value in update_log.items():
            if "reward" in tag:
                reward_dict[tag] = value
            if "action" in tag:
                continue
            else:
                self.writer.add_scalar(tag, value, i_iter)

        if 'total_reward' in reward_dict.keys():
            del reward_dict['total_reward']

        if show_r_range:
            self.writer.add_scalars('reward', reward_dict, i_iter)
        else:
            self.writer.add_scalar('avg_reward', reward_dict['avg_reward'], i_iter)

    def summary(self, msg, verbose=3):
        if verbose == 3:
            self.test_log.info(msg)
        elif verbose == 4:
            self.test_log.warning(msg)
        elif verbose <= 2:
            self.test_log.debug(msg)
        else:
            self.test_log.critical(msg)

    def write(self, msg, verbose=3):
        if verbose == 3:
            self.train_log.info(msg)
        elif verbose == 4:
            self.train_log.warning(msg)
        elif verbose <= 2:
            self.train_log.debug(msg)
        else:
            self.train_log.critical(msg)
