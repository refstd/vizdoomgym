import logging
import os
from time import sleep

import cv2
import gym
from gym import spaces
from gym.envs.classic_control import rendering
from gym.utils import seeding
import numpy as np
from vizdoom import *

log = logging.getLogger(__name__)


CONFIGS = [
    ['basic.cfg', 3],                       # 0
    ['deadly_corridor.cfg', 7],             # 1
    ['defend_the_center.cfg', 3],           # 2
    ['defend_the_line.cfg', 3],             # 3
    ['health_gathering.cfg', 3],            # 4
    ['my_way_home.cfg', 5],                 # 5
    ['predict_position.cfg', 3],            # 6
    ['take_cover.cfg', 2],                  # 7
    ['deathmatch.cfg', 20],                 # 8
    ['health_gathering_supreme.cfg', 3],    # 9
    ['my_way_home_sparse.cfg', 5],          # 10
    ['my_way_home_very_sparse.cfg', 5],     # 11
    ['my_way_home_goal.cfg', 7],            # 12
]


class VizdoomEnv(gym.Env):

    def __init__(self, level, show_automap=False):
        self.initialized = False

        # init game
        self.level = level
        self.show_automap = show_automap
        self.game = None
        self.state = None

        self.curr_seed = 0

        self.screen_w, self.screen_h, self.channels = 640, 480, 3
        self.screen_resolution = ScreenResolution.RES_640X480
        self.calc_observation_space()

        self.action_space = spaces.Discrete(CONFIGS[self.level][1])

        self.viewer = None

        self.seed()

    def calc_observation_space(self):
        self.observation_space = spaces.Box(0, 255, (self.screen_w, self.screen_h, self.channels), dtype=np.uint8)

    def _ensure_initialized(self, mode='algo'):
        if self.initialized:
            # Doom env already initialized!
            return

        self.game = DoomGame()
        scenarios_dir = os.path.join(os.path.dirname(__file__), 'scenarios')
        self.game.load_config(os.path.join(scenarios_dir, CONFIGS[self.level][0]))
        self.game.set_screen_resolution(self.screen_resolution)

        if mode == 'algo':
            self.game.set_window_visible(False)
        elif mode == 'human':
            self.game.add_game_args('+freelook 1')
            self.game.set_window_visible(True)
            self.game.set_mode(Mode.SPECTATOR)
        else:
            raise Exception('Unsupported mode')

        if self.show_automap:
            self.game.set_automap_buffer_enabled(True)
            self.game.set_automap_mode(AutomapMode.OBJECTS)
            self.game.set_automap_rotate(False)
            self.game.set_automap_render_textures(False)

            self.game.add_available_game_variable(GameVariable.POSITION_X)
            self.game.add_available_game_variable(GameVariable.POSITION_Y)
            self.game.add_available_game_variable(GameVariable.POSITION_Z)
            # self.game.add_game_args("+am_followplayer 1")
            self.game.add_game_args("+viz_am_center 1")
            self.game.add_game_args("+am_backcolor 000000")
            self.game.add_game_args("+am_showthingsprites 0")
            self.game.add_game_args("+am_cheat 0")
            # self.game.add_game_args("+am_yourcolor 0000ff")

        self.game.init()

        self.initialized = True

    def _start_episode(self):
        if self.curr_seed > 0:
            self.game.set_seed(self.curr_seed)
            self.curr_seed = 0
        self.game.new_episode()
        return

    def seed(self, seed=None):
        self.curr_seed = seeding.hash_seed(seed) % 2 ** 32
        return [self.curr_seed]

    def step(self, action):
        self._ensure_initialized()

        # convert action to vizdoom action space (one hot)
        act = np.zeros(self.action_space.n)
        act[action] = 1
        act = np.uint8(act)
        act = act.tolist()

        reward = self.game.make_action(act)
        state = self.game.get_state()
        done = self.game.is_episode_finished()
        if not done:
            observation = np.transpose(state.screen_buffer, (1, 2, 0))
        else:
            observation = np.zeros(self.observation_space.shape, dtype=np.uint8)

        return observation, reward, done, {}

    def reset(self):
        self._ensure_initialized()

        self._start_episode()
        self.state = self.game.get_state()
        img = self.state.screen_buffer
        return np.transpose(img, (1, 2, 0))

    def render(self, mode='human'):
        try:
            img = self.game.get_state().screen_buffer
            img = np.transpose(img, [1, 2, 0])

            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer(maxwidth=800)
            self.viewer.imshow(img)
        except AttributeError:
            pass

    def close(self):
        if self.viewer is not None:
            self.viewer.close()

    def play_human_mode(self):
        self._ensure_initialized('human')
        self._start_episode()

        while not self.game.is_episode_finished():
            self.game.advance_action()
            state = self.game.get_state()
            total_reward = self.game.get_total_reward()

            if state is not None:
                print('===============================')
                print('State: #' + str(state.number))
                print('Action: \t' + str(self.game.get_last_action()) + '\t (=> only allowed actions)')
                print('Reward: \t' + str(self.game.get_last_reward()))
                print('Total Reward: \t' + str(total_reward))

                if self.show_automap and state.automap_buffer is not None:
                    map = state.automap_buffer
                    map = np.swapaxes(map, 0, 2)
                    map = np.swapaxes(map, 0, 1)
                    cv2.imshow('ViZDoom Automap Buffer', map)
                    cv2.waitKey(28)
                else:
                    sleep(0.02857)  # 35 fps = 0.02857 sleep between frames

        if self.show_automap:
            cv2.destroyAllWindows()
        sleep(1)
        print('===============================')
        print('Done')
        return

    @staticmethod
    def get_keys_to_action():
        # you can press only one key at a time!
        keys = {(): 2,
                (ord('a'),): 0,
                (ord('d'),): 1,
                (ord('w'),): 3,
                (ord('s'),): 4,
                (ord('q'),): 5,
                (ord('e'),): 6}
        return keys
