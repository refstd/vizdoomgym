from vizdoomgym.envs.vizdoomenv import VizdoomEnv


class VizdoomMyWayHomeVerySparse(VizdoomEnv):

    def __init__(self, **kwargs):
        super(VizdoomMyWayHomeVerySparse, self).__init__(11, **kwargs)
