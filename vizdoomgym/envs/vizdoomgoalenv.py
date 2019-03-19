import logging

from gym import spaces

from vizdoomgym.envs import VizdoomEnv

log = logging.getLogger(__name__)


class VizdoomGoalEnv(VizdoomEnv):

    def __init__(self, level):
        super(VizdoomGoalEnv, self).__init__(level)
        self._calc_observation_space_with_goal()

        self.goal_obs = None

    def _calc_observation_space_with_goal(self):
        """We have two observations: one is the current observation and one is the goal room observation."""
        current_obs_space = self.observation_space
        self.observation_space = spaces.Dict({'obs': current_obs_space, 'goal': current_obs_space})

    def _make_observation(self, obs):
        return {'obs': obs, 'goal': self.goal_obs}

    def reset(self):
        """
        First frame in the episode gives us the goal location, then we're teleported to the actual starting
        location from which we can start exploring.
        """
        self.goal_obs = super(VizdoomGoalEnv, self).reset()

        while True:
            obs, _, _, _ = super(VizdoomGoalEnv, self).step(0)
            state = self.game.get_state()
            variables = state.game_variables
            goal_obs_generated = variables[-1]
            if goal_obs_generated >= 1:
                break

        return self._make_observation(obs)

    def step(self, action):
        obs, reward, done, info = super(VizdoomGoalEnv, self).step(action)
        return self._make_observation(obs), reward, done, info


class VizDoomMyWayHomeGoal(VizdoomGoalEnv):
    def __init__(self):
        super(VizDoomMyWayHomeGoal, self).__init__(12)
