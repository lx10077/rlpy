import gym
from collections import deque
from gym.spaces.box import Box
from envs.helper_env import *


# ====================================================================================== #
# Inheritance of gym.Wrapper
# ====================================================================================== #
class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """Do no-op action for a number of steps in [1, noop_max]. """
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)  # pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing. """
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to batch_to_var bonus lives
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            # for Qbert somtimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame. """
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype='uint8')
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations. """
        total_reward = 0.0
        done = None
        info = {}
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2: 
                self._obs_buffer[0] = obs
            if i == self._skip - 1: 
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.

        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = Box(low=0.0, high=255.0,
                                     shape=(shp[0], shp[1], shp[2] * k),
                                     dtype=np.float32)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info


# ====================================================================================== #
# Inheritance of gym.RewardWrapper
# ====================================================================================== #
class ClipRewardEnv(gym.RewardWrapper):
    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign. """
        return np.sign(reward)


# ====================================================================================== #
# Inheritance of gym.ObservationWrapper
# ====================================================================================== #
class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env):
        """Warp frames to 80x80, scaled and cropped based on configuration. """
        super(WarpFrame, self).__init__(env)
        self.observation_space = Box(low=0.0, high=1.0, shape=(80, 80, 1), dtype=np.float32)
        self.conf = None
        for name, conf in conf_dict.items():
            if name in env.spec.id:
                self.conf = conf
                break
        if self.conf is None:
            print("Frame configuration of {:s} isn't set! Use default.".format(env.spec.id))
            self.conf = conf_dict["Default"]

    def observation(self, observation):
        out = process_frame80(observation, self.conf)
        return out[:, :, None]


class ScaledFloatFrame(gym.ObservationWrapper):
    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0


class NormalizedEnv(gym.ObservationWrapper):
    def __init__(self, env):
        super(NormalizedEnv, self).__init__(env)
        self.state_mean = 0
        self.state_std = 0
        self.alpha = 0.9999
        self.num_steps = 0

    def observation(self, observation):
        self.num_steps += 1
        self.state_mean = self.state_mean * self.alpha + observation.mean() * (1 - self.alpha)
        self.state_std = self.state_std * self.alpha + observation.std() * (1 - self.alpha)

        unbiased_mean = self.state_mean / (1 - pow(self.alpha, self.num_steps))
        unbiased_std = self.state_std / (1 - pow(self.alpha, self.num_steps))

        return (observation - unbiased_mean) / (unbiased_std + 1e-8)


class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        """Change image shape from WHC to CWH. """
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = Box(low=0.0, high=1.0,
                                     shape=(old_shape[-1], old_shape[0], old_shape[1]),
                                     dtype=np.float32)

    def observation(self, observation):
        return np.swapaxes(observation, 2, 0)


# ====================================================================================== #
# Atari wrapper-making functions
# ====================================================================================== #
def make_atari(env_id):
    env = gym.make(env_id)
    assert 'NoFrameskip' in env.spec.id
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    return env


def make_normalized_atari(env, episode_life=True, clip_rewards=True,
                          frame_stack=False, pytorch_img=False):
    """Configure normalized and scaled environment. """
    if 'NoFrameskip' in env.spec.id:
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
    if episode_life:
        env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)  # FloatFrame has been scaled here.
    env = NormalizedEnv(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, 4)
    if pytorch_img:
        env = ImageToPyTorch(env)
    return env


def wrap_deepmind(env, episode_life=True, clip_rewards=True, 
                  frame_stack=False, pytorch_img=False):
    """Configure environment for DeepMind-style Atari. """
    if episode_life:
        env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)  # FloatFrame isn't scaled here.
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, 4)
    if pytorch_img:
        env = ImageToPyTorch(env)
    return env
