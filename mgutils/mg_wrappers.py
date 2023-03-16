from typing import Optional, Any, Tuple, List

import gymnasium as gym
#from gymnasium.core import Wrapper
from .mg_asp import get_facts_from_minigrid

class AccumScore(gym.Wrapper):
    """
    Wrapper which adds a .score property, by accumulating step rewards

    Example:
        >>> import minigrid
        >>> import gymnasium as gym
        >>> from minigrid.wrappers import ActionBonus
        >>> from mg_utils.wrappers import AccumScore
        >>> env = gym.make("MiniGrid-Empty-5x5-v0")
        >>> env_bonus = ActionBonus(env)
        >>> env_score = AccumScore(env_bonus)
        >>> _, _ = env_bonus.reset(seed=0)
        >>> _, reward, _, _, _ = env_bonus.step(1)
        >>> print(reward)
        1.0
        >>> _, reward, _, _, _ = env_bonus.step(1)
        >>> print(reward)
        1.0
        >>> print(env_score.score)
        2.0
    """

    def __init__(self, env):
        """A wrapper that adds an exploration bonus to less visited (state,action) pairs.

        Args:
            env: The environment to apply the wrapper
        """
        super().__init__(env)
        self.score = 0.0

    def step(self, action):
        """Steps through the environment with `action`."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        if reward:
            self.score += float(reward)
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Resets the environment with `kwargs`."""
        self.score = 0.0
        return self.env.reset(**kwargs)


class SymbolicFactsWrapper(gym.Wrapper):
    """
    Wrapper which returns a list of symbolic facts representing agent observations

    Example:
        >>> import minigrid
        >>> import gymnasium as gym
        >>> env = gym.make("MiniGrid-Empty-5x5-v0")
        >>> env = FactsWrapper(env)
        >>> _, _ = env.reset(seed=0)
        >>> _, reward, _, _, infos = env_bonus.step(1)
        >>> print(infos['facts'])
    """

    def __init__(self, env):
        """A wrapper that exposes minigrid state as a list of symbolic facts.

        Args:
            env: The environment to apply the wrapper
        """
        super().__init__(env)

    def _add_facts_to_info(self, info):
        if not 'facts' in info:
            info['facts'] = []
        static_facts_list, fluent_facts_list = get_facts_from_minigrid(self.env)
        facts_list = static_facts_list + fluent_facts_list
        info['facts'].append(facts_list)
        return info

    def _map_object_ids(self, info, facts_list, t:int):
        if not 'facts' in info:
            info['facts'] = []
        obj_infos = assign_obj_ids_for_ASP(self.env)

        info['facts'].append(facts_list)
        return info

    def step(self, action):
        """Steps through the environment with `action`."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._add_facts_to_info(info)
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Resets the environment with `kwargs`."""
        obs, info = self.env.reset(**kwargs)
        self._map_obj_ids()
        self._add_facts_to_info(info)
        return obs, info

    # copyied from SymbolicObsWrapper
    # def observation(self, obs):
    #     objects = np.array(
    #         [OBJECT_TO_IDX[o.type] if o is not None else -1 for o in self.grid.grid]
    #     )
    #     agent_pos = self.env.agent_pos
    #     w, h = self.width, self.height
    #     grid = np.mgrid[:w, :h]
    #     grid = np.concatenate([grid, objects.reshape(1, w, h)])
    #     grid = np.transpose(grid, (1, 2, 0))
    #     grid[agent_pos[0], agent_pos[1], 2] = OBJECT_TO_IDX["agent"]
    #     obs["image"] = grid

    #     return obs
