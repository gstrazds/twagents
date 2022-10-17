# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

from textworld import Agent
from symbolic.wrappers.gym_wrappers import TWoWrapper


class WalkthroughDone(NameError):
    pass


class TwOracleAgent(Agent):
    """ Agent that uses TWOracle to play a TextWorld game. """

    def __init__(self, commands=None):
        self.commands = commands

    @property
    def wrappers(self):
        return [TWoWrapper]

    def reset(self, env):
        env.display_command_during_render = True
        if self.commands is not None:
            self._commands = iter(self.commands)
            return  # Commands already specified.

        game_state = env.reset()
        if game_state.get("extra.walkthrough") is None:
            msg = "WalkthroughAgent is only supported for games that have a walkthrough."
            raise NameError(msg)

        # Load command from the generated game.
        self._commands = iter(game_state.get("extra.walkthrough"))

    def act(self, game_state, reward, done):
        try:
            action = next(self._commands)
        except StopIteration:
            raise WalkthroughDone()

        action = action.strip()  # Remove trailing \n, if any.
        return action
