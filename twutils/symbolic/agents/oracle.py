# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

from textworld import Agent, Environment
from symbolic.wrappers.gym_wrappers import TWoWrapper


class WalkthroughDone(NameError):
    pass


class TwOracleAgent(Agent):
    """ Agent that uses TWOracle to play a TextWorld game. """

    def __init__(self, commands=None, **kwargs):
        super().__init__(**kwargs)
        self.commands = commands
        # self.env = None

    @property
    def wrappers(self):
        return [TWoWrapper]

    def reset(self, env: Environment):
        print(f"TwOracleAgent({self}).reset({env})")
        # self.env = env
        env.display_command_during_render = True

        print(env, env.reset)
        game_state = env.reset()
        game_state.next_command = "do something"
        #print(game_state)
        if not hasattr(game_state, "next_command"):
            msg = "TwOracleAgent only works for games with a dynamic next_command property (e.g. from TWoWrapper)"
            raise NameError(msg)

        if self.commands is not None:
            self._commands = iter(self.commands)
            return  # Commands already specified.
        # Load command from the generated game.
        if not hasattr(game_state, "extra.walkthrough"):
            msg = "TwOracleAgent only works for games with extra.walkthrough property"
            print(game_state)
            raise NameError(msg)
        else:
            self._commands = iter(game_state.get("extra.walkthrough"))

    def act(self, game_state, reward, done):
        command = game_state.get("next_command", "NO NEXT COMMAND FOUND")
        print("next_command =", command)
        if self._commands:
            try:
                action = next(self._commands)
                print("action from walkthrough =", action)
                command = action
            except StopIteration:
                raise WalkthroughDone()

        if not command:
            raise WalkthroughDone()
        command = command.strip()  # Remove trailing \n, if any.
        return command  # will be followed by a call to env.step(command)
