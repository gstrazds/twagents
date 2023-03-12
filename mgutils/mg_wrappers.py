import gymnasium as gym
#from gymnasium.core import Wrapper

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
