from vizdoomgym.envs.vizdoomenv import VizdoomEnv


class VizdoomMyWayHomeSparse(VizdoomEnv):

    def __init__(self, **kwargs):
        super(VizdoomMyWayHomeSparse, self).__init__(10, **kwargs)
