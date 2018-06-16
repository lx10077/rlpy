import numpy as np
import json


# ====================================================================================== #
# Configuration
# ====================================================================================== #
class Cfg(object):
    def __init__(self, name="default", config_file=None, parse=None, init_dict=None, **kwargs):
        """A class to handle all configurations
        """
        self.name = name
        self.config_file = config_file
        self.config = {}
        self.config.update(kwargs)
        if init_dict is not None:
            self.read_dict(init_dict)
        if parse is not None:
            self.read_parse(parse)

    def set_saved_file(self, config_file):
        self.config_file = config_file

    def keys(self):
        return self.config.keys()

    def read_parse(self, parse):
        parse_dict = parse.__dict__
        self.config.update(parse_dict)

    def read_dict(self, input_dict):
        self.config.update(input_dict)

    def save_config(self, file=None):
        try:
            if file is None:
                if self.config_file is not None:
                    file = self.config_file
                else:
                    raise ValueError('No dir to save.')
            else:
                if self.config_file is None:
                    self.set_saved_file(file)
            with open(file, "w") as f:
                json.dump(self.config, f)
        except Exception as e:
            print("[Error]     Fail to save the file {}.".format(file))
            raise Exception(e)

    def load_config(self, file=None):
        try:
            if file is None:
                if self.config_file is not None:
                    file = self.config_file
                else:
                    raise ValueError('No dir to load.')
            with open(file, "r") as f:
                self.config = json.load(f)
        except Exception as e:
            print("[Error]     Fail to open the file {}.".format(file))
            raise Exception(e)

    def __getitem__(self, item):
        try:
            return self.config[item]
        except Exception as e:
            print("[Error]     No such item {} in config.".format(str(item)))
            raise Exception(e)

    def __setitem__(self, key, value):
        if key in self.config:
            print("[Warning]   {} has been in the config. Overwrite it now.".format(key))
        self.config[key] = value

    def __delitem__(self, key):
        try:
            del self.config[key]
        except Exception as e:
            print("[Error]     Fail to delete the key {}.".format(key))
            print(Exception(e))

    def __len__(self):
        return len(self.config)

    def __contains__(self, item):
        if item in self.config:
            return True
        else:
            return False

    def __repr__(self):
        return self.config.__repr__()


# ====================================================================================== #
# Statistic calculation (used in vector-state environments)
# ====================================================================================== #
class RunningStat(object):
    def __init__(self, shape):
        """Running estimates of mean, std of env states.

        -   _n = number of seen states
        -   _M = mean of env states
        -   _S = std of env states
        """
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)

    def state_dict(self):
        return {"n": self._n, "M": self._M, "S": self._S}

    def set_state(self, state_dict):
        self._n = state_dict["n"]
        self._M = state_dict["M"]
        self._S = state_dict["S"]

    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            old_m = self._M.copy()
            self._M[...] = old_m + (x - old_m) / self._n
            self._S[...] = self._S + (x - old_m) * (x - self._M)

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape


class ZFilter(object):
    def __init__(self, shape, demean=True, destd=True, clip=10.0):
        """Normalize env states with running estimates.

        -   y = (x - mean) / std
        """
        self.demean = demean
        self.destd = destd
        self.clip = clip

        self.rs = RunningStat(shape)

    def __call__(self, x, update=True):
        if update:
            self.rs.push(x)
        if self.demean:
            x = x - self.rs.mean
        if self.destd:
            x = x / (self.rs.std + 1e-8)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x

    def save_dict(self):
        return {"demean": self.demean, "destd": self.destd, "clip": self.clip}

    def set_state(self, state_dict):
        self.demean = state_dict["demean"]
        self.destd = state_dict["destd"]
        self.clip = state_dict["clip"]

    def state_dict(self):
        return {"Zfilter": self.save_dict(),
                "RuningStat": self.rs.state_dict()}

    def load_state_dict(self, state_dict):
        if "Zfilter" in state_dict:
            print("[Load]     Load Zfilter...")
            self.set_state(state_dict["Zfilter"])
        if "RuningStat" in state_dict:
            print("[Load]     Load RunningStat...")
            self.rs.set_state(state_dict["RuningStat"])
