import os
from collections import deque
from envs.helper_env import *
from ale_python_interface import ALEInterface


class AleEnv(object):
    def __init__(self, rom_file, frame_skip, num_frames, frame_size,
                 no_op_start, rand_seed, dead_as_eoe):
        self.ale = self._init_ale(rand_seed, rom_file)
        # normally (160, 210)
        self.actions = self.ale.getMinimalActionSet()

        self.frame_skip = frame_skip
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.no_op_start = no_op_start
        self.dead_as_eoe = dead_as_eoe

        self.clipped_reward = 0
        self.total_reward = 0
        screen_width, screen_height = self.ale.getScreenDims()
        self.prev_screen = np.zeros(
            (screen_height, screen_width, 3), dtype=np.float32)
        self.frame_queue = deque(maxlen=num_frames)
        self.end = True

    @staticmethod
    def _init_ale(rand_seed, rom_file):
        assert os.path.exists(rom_file), '%s does not exists.'
        ale = ALEInterface()
        ale.setInt('random_seed', rand_seed)
        ale.setBool('showinfo', False)
        ale.setInt('frame_skip', 1)
        ale.setFloat('repeat_action_probability', 0.0)
        ale.setBool('color_averaging', False)
        ale.loadROM(rom_file)
        return ale

    @property
    def num_actions(self):
        return len(self.actions)

    def _get_current_frame(self):
        # global glb_counter
        screen = self.ale.getScreenRGB()
        max_screen = np.maximum(self.prev_screen, screen)
        frame = preprocess_frame(max_screen, self.frame_size)
        frame /= 255.0
        # cv2.imwrite('test_env/%d.png' % glb_counter, cv2.resize(frame, (800, 800)))
        # glb_counter += 1
        # print('glb_counter', glb_counter)
        return frame

    def reset(self):
        for _ in range(self.num_frames - 1):
            self.frame_queue.append(
                np.zeros((self.frame_size, self.frame_size), dtype=np.float32))

        self.ale.reset_game()
        self.clipped_reward = 0
        self.total_reward = 0
        self.prev_screen = np.zeros(self.prev_screen.shape, dtype=np.float32)

        n = np.random.randint(0, self.no_op_start)
        for i in range(n):
            if i == n - 1:
                self.prev_screen = self.ale.getScreenRGB()
            self.ale.act(0)

        self.frame_queue.append(self._get_current_frame())
        assert not self.ale.game_over()
        self.end = False
        return np.array(self.frame_queue)

    def step(self, action_idx):
        assert not self.end
        reward = 0
        clipped_reward = 0
        old_lives = self.ale.lives()

        for _ in range(self.frame_skip):
            self.prev_screen = self.ale.getScreenRGB()
            r = self.ale.act(self.actions[action_idx])
            reward += r
            clipped_reward += np.sign(r)
            dead = (self.ale.lives() < old_lives)
            if self.ale.game_over() or (self.dead_as_eoe and dead):
                self.end = True
                break

        self.frame_queue.append(self._get_current_frame())
        self.total_reward += reward
        self.clipped_reward += clipped_reward
        return np.array(self.frame_queue), clipped_reward, self.end

    def render(self):
        cv2.imshow('screen', self.prev_screen)
        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()
