from collections import namedtuple

from rlpyt.envs.base import Env, EnvStep
from rlpyt.spaces.int_box import IntBox
from rlpyt.utils.quick_args import save__init__args
from rlpyt.samplers.collections import TrajInfo


EnvInfo = namedtuple("EnvInfo", ["game_score", "traj_done"])


class FtwcTrajInfo(TrajInfo):
    """TrajInfo class for use with Ftwc/Qait Env, stores raw game score in addition to reward signal.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.GameScore = 0

    def step(self, observation, action, reward, done, agent_info, env_info):
        super().step(observation, action, reward, done, agent_info, env_info)
        self.GameScore += getattr(env_info, "game_score", 0)


class FtwcEnv(Env):
    """Wraps/adapts QaitOracleEnv for rlpyt .

    Output `env_info` includes:
        * `game_score`: raw game score, separate from reward clipping.
        * `traj_done`: special signal which signals game-over or timeout, so that sampler doesn't reset the environment when ``done==True`` but ``traj_done==False``, which can happen when ``episodic_lives==True``.

The action space is an `IntBox` for the number of actions.  The observation
space is an `IntBox` with ``dtype=uint8`` to save memory; conversion to float
should happen inside the agent's model's ``forward()`` method.

Args:
    game (str): game name
    episodic_lives (bool): if ``True``, output ``done=True`` but ``env_info[traj_done]=False`` when an episode ends
    max_start_noops (int): upper limit for random number of noop actions after reset
    horizon (int): max number of steps before timeout / ``traj_done=True``
"""


def __init__(self,
             game="pong",
             episodic_lives=True,
             fire_on_reset=False,
             max_start_noops=30,
             horizon=27000,
             ):
        save__init__args(locals(), underscore=True)

        self._action_space = IntBox(low=0, high=len(self._action_set))
        # obs_shape = (num_img_obs, H, W)
        self._observation_space = IntBox(low=0, high=255, shape=obs_shape, dtype="uint32")
        self._max_frame = self.ale.getScreenGrayscale()
        self._raw_frame_1 = self._max_frame.copy()
        self._raw_frame_2 = self._max_frame.copy()
        self._obs = np.zeros(shape=obs_shape, dtype="uint8")

        self._horizon = int(horizon)
        self.reset()

    def reset(self):
        """Performs hard reset of ALE game."""
        # self.ale.reset_game()
        # self._reset_obs()
        # self._life_reset()
        # for _ in range(np.random.randint(0, self._max_start_noops + 1)):
        #     self.ale.act(0)
        # if self._fire_on_reset:
        #     self.fire_and_up()
        # self._update_obs()  # (don't bother to populate any frame history)
        self._step_counter = 0
        return self.get_obs()

    def step(self, action):
        a = self._action_set[action]
        game_score = np.array(0., dtype="float32")
        for _ in range(self._frame_skip - 1):
            game_score += self.ale.act(a)
        self._get_screen(1)
        game_score += self.ale.act(a)
        lost_life = self._check_life()  # Advances from lost_life state.
        if lost_life and self._episodic_lives:
            self._reset_obs()  # Internal reset.
        self._update_obs()
        reward = np.sign(game_score) if self._clip_reward else game_score
        game_over = self.ale.game_over() or self._step_counter >= self.horizon
        done = game_over or (self._episodic_lives and lost_life)
        info = EnvInfo(game_score=game_score, traj_done=game_over)
        self._step_counter += 1
        return EnvStep(self.get_obs(), reward, done, info)

    def render(self, wait=10, show_full_obs=False):
        super().render()
        # raise NotImplementedError
