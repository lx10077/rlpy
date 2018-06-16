from utils.torch import *
from core.logger import Logger
import time
import os


class ActorCriticTrainer(object):
    def __init__(self, agent, updater, cfg, evaluator=None):
        self.agent = agent
        self.id = self.agent.id
        self.updater = updater
        self.evaluator = evaluator
        self.cfg = cfg
        self.log = Logger(self.id, self.cfg)
        self.model_dir = self.log.task.model_dir
        self.iter_i = 0
        if self.evaluator is not None:
            self.evaluator.set_logger(self.log)

        self.gpu = cfg["gpu"] if "gpu" in cfg else False
        self.tau = cfg["tau"]
        self.gamma = cfg["gamma"]
        self.min_batch_size = cfg["min_batch_size"]
        self.max_iter_num = cfg["max_iter_num"]
        self.min_batch_size = cfg["min_batch_size"]
        self.log_interval = cfg["log_interval"]
        self.save_model_interval = cfg["save_model_interval"]
        self.eval_model_interval = cfg["eval_model_interval"]
        self.begin_i = 0

    def start(self, every_save=False):
        while self.iter_i < self.max_iter_num:
            batch, train_log = self.agent.collect_samples(self.min_batch_size)
            batch = self.agent.batch2tensor(batch)
            t0 = time.time()
            train_log = self.updater(batch, train_log, self.iter_i)
            t1 = time.time()
            train_log["update_time"] = t1 - t0

            self.log.record(self.iter_i, train_log)

            if self.iter_i % self.log_interval == 0:
                msg = '{}\tT_sample {:.4f}\tT_update {:.4f}\tR_min {:.2f}\tR_max {:.2f}\tR_avg {:.2f}'.format(
                    self.iter_i, train_log['sample_time'], t1 - t0, train_log['min_reward'],
                    train_log['max_reward'], train_log['avg_reward'])
                self.log.write(msg, 3)

            if self.evaluator is not None and self.eval_model_interval > 0 \
                    and (self.iter_i + 1) % self.eval_model_interval == 0:
                test_log = self.evaluator.eval(self.iter_i)
                msg = 'Test in {}\tR_min {:.2f}\tR_max {:.2f}\tR_avg {:.2f}'.format(
                    self.iter_i, min(test_log['rs']), max(test_log['rs']), test_log['ravg'])
                self.log.summary(msg, 3)

            if self.save_model_interval > 0 and (self.iter_i + 1) % self.save_model_interval == 0:
                self.save_checkpoint('iter-' + str(self.iter_i))

            if every_save:
                self.save_checkpoint('latest')
            self.iter_i += 1

        self.log.write('Complete training: {:d} -> {:d}.'.format(self.begin_i, self.max_iter_num), 3)

    def save_checkpoint(self, name):
        self.log.write('Saving the {} checkpoint...'.format(name), 3)
        file = os.path.join(self.model_dir, name)
        save_dict = {}
        for model, net in self.agent.model_dict.items():
            save_dict[model] = net.state_dict()
        if self.agent.running_state is not None:
            save_dict["running_state"] = self.agent.running_state.save_dict()
        save_dict["iter_i"] = self.iter_i
        try:
            torch.save(save_dict, file)
        except Exception as e:
            self.log.write('Fail to save {}: '.format(file), 4)
            raise Exception(e)

    def load_checkpoint(self, name="latest"):
        self.log.write('Loading the {} checkpoint...'.format(name), 3)
        file = os.path.join(self.model_dir, name)
        try:
            save_dict = get_state_dict(file)
            self.iter_i = save_dict["iter_i"]
            del save_dict["iter_i"]
            if "running_state" in save_dict and self.agent.running_state:
                self.agent.running_state.load(save_dict["running_state"])
                del save_dict["running_state"]
            for model, net in self.agent.model_dict.items():
                state_dict = net.state_dict()
                keys = list(state_dict.keys())
                for key in keys:
                    try:
                        state_dict[key] = save_dict[model][key]
                    except KeyError:
                        self.log.write("{}'s {} isn't in the save dict".format(model, key), 5)
                        continue
                net.load_state_dict(state_dict)
        except Exception as e:
            self.log.write("Fail to open {}.".format(file), 4)
            raise Exception(e)
