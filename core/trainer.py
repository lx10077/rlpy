from utils.tools import *
from utils.torch import *
import pickle


class ActorCriticTrainer(object):
    def __init__(self, agent, updater, cfg, evaluator=None):
        self.agent = agent
        self.updater = updater
        self.evaluator = evaluator
        self.cfg = cfg
        self.asset_dir = assets_dir()
        self.model_dir = set_dir(self.asset_dir, "learned_models")
        self.config_dir = set_dir(self.asset_dir, "config")
        self.id = self.agent.id
        self.cfg.set_saved_file(self.config_dir + "/" + self.id + "-config.pkl")
        self.cfg.save_config()

        self.gpu = cfg["gpu"] if "gpu" in cfg else False
        self.tau = cfg["tau"]
        self.gamma = cfg["gamma"]
        self.min_batch_size = cfg["min_batch_size"]
        self.max_iter_num = cfg["max_iter_num"]
        self.min_batch_size = cfg["min_batch_size"]
        self.log_interval = cfg["log_interval"]
        self.save_model_interval = cfg["save_model_interval"]
        self.eval_model_interval = cfg["eval_model_interval"]
        self.begin_i = -1
        self.meta_info = {}

    def setup(self, file=None):
        self.load_model(file=file)
        self.load_meta_info()

    def start(self):
        for iter_i in range(self.begin_i + 1, self.max_iter_num):
            batch, log = self.agent.collect_samples(self.min_batch_size)
            batch = self.agent.batch2tensor(batch)
            t0 = time.time()
            log = self.updater(batch, log)
            t1 = time.time()
            log["update_time"] = t1 - t0

            if self.evaluator is not None:
                self.evaluator.record(iter_i, log)
                self.evaluator.monitor(iter_i, log)

            if iter_i % self.log_interval == 0:
                print('{}\tT_sample {:.4f}\tT_update {:.4f}\tR_min {:.2f}\tR_max {:.2f}\tR_avg {:.2f}'.format(
                    iter_i, log['sample_time'], t1 - t0, log['min_reward'], log['max_reward'], log['avg_reward']))

            if self.save_model_interval > 0 and (iter_i + 1) % self.save_model_interval == 0:
                self.save_model()

            if self.evaluator is not None and self.eval_model_interval > 0 \
                    and (iter_i + 1) % self.eval_model_interval == 0:
                log = self.evaluator.eval(iter_i)
                self.meta_info["test_ %d" % iter_i] = log

        print("[Finish] Complete training: {:d} -> {:d}.".format(self.begin_i + 1, self.max_iter_num))
        if self.evaluator:
            self.evaluator.save_records()
        self.save_meta_info()

    def save_meta_info(self):
        print("[Save] Saving the meta information...")
        file = self.model_dir + '/' + self.id + "-metadata.pkl"
        try:
            with open(file, "wb") as f:
                pickle.dump(self.meta_info, f)
        except Exception as e:
            print("[Error] Fail to save the meta information.")
            print(Exception(e))
            pass

    def load_meta_info(self):
        print("[Load] Saving the meta information...")
        file = self.model_dir + '/' + self.id + "-metadata.pkl"
        try:
            with open(file, "wr") as f:
                self.meta_info = pickle.load(f)
        except Exception as e:
            print("[Error] Fail to load the desired meta information.")
            print(Exception(e))
            pass

    def save_model(self, file=None):
        print("[Save] Saving the learned model...")
        if file is None:
            file = os.path.join(self.model_dir, self.id + ".pth")
        save_dict = {}
        for model, net in self.agent.model_dict.items():
            save_dict[model] = net.state_dict()
        if self.agent.running_state is not None:
            save_dict["running_state"] = self.agent.running_state.save_dict()
        if self.begin_i > 0:
            save_dict["begin_i"] = self.begin_i
        try:
            torch.save(save_dict, file)
        except Exception as e:
            print("[Error] Fail to save {}.".format(file))
            print(Exception(e))
            pass

    def load_model(self, skip=None, file=None):
        print("[Load] Loading saved model...")
        if file is None:
            file = os.path.join(self.model_dir, self.id + ".pth")
        try:
            save_dict = get_state_dict(file)
            if "running_state" in save_dict and self.agent.running_state:
                self.agent.running_state.load(save_dict["running_state"])
                del save_dict["running_state"]
            if "begin_i" in save_dict:
                self.begin_i = save_dict["begin_i"]
                del save_dict["begin_i"]
            for model, net in self.agent.model_dict.items():
                state_dict = net.state_dict()
                keys = list(state_dict.keys())
                for key in keys:
                    if skip and any(s in key for s in skip):
                        continue
                    try:
                        state_dict[key] = save_dict[model][key]
                    except KeyError:
                        print("[KeyError] {}'s {} isn't in the save dict".format(model, key))
                        continue
                net.load_state_dict(state_dict)
        except Exception as e:
            print("[Error] Fail to open {}.".format(file))
            print(Exception(e))
            pass
