import numpy as np
import cv2


# ====================================================================================== #
# Atari frame processing function
# ====================================================================================== #
class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not believe how complex the previous solution was.
        """
        self._frames = frames

    def __array__(self, dtype=None):
        out = np.concatenate(self._frames, axis=2)
        if dtype is not None:
            out = out.astype(dtype)
        return out


def process_frame80(frame, conf):
    frame = frame[conf["crop1"]:conf["crop2"] + 160, :160]
    frame = frame.mean(2)
    frame = frame.astype(np.float32)
    frame *= (1.0 / 255.0)
    frame = cv2.resize(frame, (80, conf["dimension2"]))
    frame = cv2.resize(frame, (80, 80))
    return frame


def preprocess_frame(observ, output_size, interpolation=cv2.INTER_AREA):
    gray = cv2.cvtColor(observ, cv2.COLOR_RGB2GRAY)
    output = cv2.resize(gray, (output_size, output_size), interpolation=interpolation)
    output = output.astype(np.float32, copy=False)
    return output


def rgb2gray(rgb):
    gray_image = 0.2126 * rgb[..., 0]
    gray_image[:] += 0.0722 * rgb[..., 1]
    gray_image[:] += 0.7152 * rgb[..., 2]
    return gray_image


def rgb2y(rgb):
    y_image = 0.299 * rgb[..., 0]
    y_image[:] += 0.587 * rgb[..., 1]
    y_image[:] += 0.114 * rgb[..., 2]
    return y_image


def cv2scale(image, hei_image, wid_image):
    return cv2.resize(image, (wid_image, hei_image), interpolation=cv2.INTER_LINEAR)


def one_hot_np(n_classes, labels):
    # This functions is different from the same-name function in utils_torch.py
    # To distinguish them, a suffix 'np' is added.
    one_hot_labels = np.zeros(labels.shape + (n_classes,))
    for c in range(n_classes):
        one_hot_labels[labels == c, c] = 1
    return one_hot_labels


# ====================================================================================== #
# Atari game frame configuration dict
# ====================================================================================== #
conf_dict = {"Default":   {"crop1": 34, "crop2": 34, "dimension2": 80},
             "Asteroids": {"crop1": 16, "crop2": 34, "dimension2": 94},
             "BeamRider": {"crop1": 20, "crop2": 20, "dimension2": 80},
             "Breakout":  {"crop1": 34, "crop2": 34, "dimension2": 80},
             "Centipede": {"crop1": 36, "crop2": 56, "dimension2": 90},
             "MsPacman":  {"crop1": 2,  "crop2": 10, "dimension2": 84},
             "Pong":      {"crop1": 34, "crop2": 34, "dimension2": 80},
             "Seaquest":  {"crop1": 30, "crop2": 30, "dimension2": 80},
             "Qbert":     {"crop1": 12, "crop2": 40, "dimension2": 94},
             "Boxing":    {"crop1": 30, "crop2": 30, "dimension2": 80},
             "SpaceInvaders": {"crop1": 8,  "crop2": 36, "dimension2": 94},
             "VideoPinball":  {"crop1": 42, "crop2": 60, "dimension2": 89}
             }
