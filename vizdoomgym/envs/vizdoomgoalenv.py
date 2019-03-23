import logging

from gym import spaces
import numpy as np

from vizdoomgym.envs import VizdoomEnv

log = logging.getLogger(__name__)


class VizdoomGoalEnv(VizdoomEnv):

    def __init__(self, level, **kwargs):
        super(VizdoomGoalEnv, self).__init__(level, **kwargs)
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
            goal_obs_generated = variables[-1]  # 'ready' user variable
            if goal_obs_generated >= 1:
                break

        return self._make_observation(obs)

    def step(self, action):
        obs, reward, done, info = super(VizdoomGoalEnv, self).step(action)

        if not done:
            pos = info['pos']

            # Check closeness with goal position
            position_close = np.abs(pos['agent_x'] - pos['goal_x']) < 20. and np.abs(pos['agent_y'] - pos['goal_y']) < 20.

            # https://gamedev.stackexchange.com/a/4472
            angle_close = 180 - np.abs(np.abs(pos['agent_a'] - pos['goal_a']) - 180) < 20.

            if position_close and angle_close:
                done = True
                reward += 10.0

        return self._make_observation(obs), reward, done, info

    def get_info(self):
        return {'pos': self.get_positions()}

    def _get_positions(self, variables):
        return {'agent_x': variables[1], 'agent_y': variables[2], 'agent_a': variables[3], 'goal_x': variables[4], 'goal_y': variables[5], 'goal_a': variables[6]}


class VizDoomMyWayHomeGoal(VizdoomGoalEnv):
    def __init__(self, **kwargs):
        super(VizDoomMyWayHomeGoal, self).__init__(12, **kwargs)


class VizDoomSptmBattleNavigation(VizdoomGoalEnv):
    def __init__(self, **kwargs):
        super(VizDoomSptmBattleNavigation, self).__init__(13, **kwargs)
