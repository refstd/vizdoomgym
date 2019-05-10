import logging

from gym import spaces
import numpy as np

from vizdoomgym.envs import VizdoomEnv

log = logging.getLogger(__name__)

class VizdoomRandomMapEnv(VizdoomEnv):
    def __init__(self, level, num_levels, **kwargs):
        super(VizdoomRandomMapEnv, self).__init__(level, **kwargs)
        self.num_levels = num_levels
        self._pick_random_map()

    def reset(self, mode='algo'):
        if self.initialized:
            self.game.close()
            self.initialized = False
        self._pick_random_map()
        obs = super(VizdoomRandomMapEnv, self).reset(mode)
        return obs

    def _pick_random_map(self):
        map_idx = self.rng.random_integers(1, self.num_levels)
        self.level_map = 'map{:02d}'.format(map_idx)


class VizdoomTexturedMazeNoGoalRandom(VizdoomRandomMapEnv):
    def __init__(self, **kwargs):
        super().__init__(
            25, num_levels=200, coord_limits=(0, 0, 2336, 2369), **kwargs)