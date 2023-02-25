#!/usr/bin/env python

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.


import argparse
import itertools

import textworld
import textworld.agents

from symbolic.agents.oracle import TwOracleAgent
from symbolic.agents.oracle import TwAnswerSetAgent

def build_parser():
    description = "Play a TextWorld game (.z8 or .ulx)."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("game")
    parser.add_argument("--mode", default="oracle", metavar="MODE",
                        choices=["oracle", "asp", "random", "human", "random-cmd", "walkthrough"],
                        help="Select an agent to play the game: %(choices)s."
                             " Default: %(default)s.")
    parser.add_argument("--asp", action="store_true",
                        help="Modifies modo=walkthrough to use clingo to generate the walkthrough")
    parser.add_argument("--max-steps", type=int, default=0, metavar="STEPS",
                        help="Limit maximum number of steps.")
    parser.add_argument("--viewer", metavar="PORT", type=int, nargs="?", const=6070,
                        help="Start web viewer.")
    parser.add_argument("--hint", action="store_true",
                        help="Display the oracle trajectory leading to winning the game.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose mode.")
    parser.add_argument("-vv", "--very-verbose", action="store_true",
                        help="Print debug information.")
    return parser


def main():
    args = build_parser().parse_args()
    if args.very_verbose:
        args.verbose = args.very_verbose

    if args.mode == "random":
        agent = textworld.agents.NaiveAgent()
    elif args.mode == "random-cmd":
        agent = textworld.agents.RandomCommandAgent()
    elif args.mode == "human":
        agent = textworld.agents.HumanAgent(oracle=args.hint)
    elif args.mode == 'walkthrough':
        if args.asp:
            from twutils.tw_asp_runner import run_gamefile
            commands = run_gamefile(args.game)
            #commands = ["examine cookbook", "prepare meal", "eat meal"]
        else:
            commands = None
        agent = textworld.agents.WalkthroughAgent(commands=commands)
    elif args.mode == 'oracle':
        agent = TwOracleAgent()
    elif args.mode == 'asp':
        agent = TwAnswerSetAgent()

    env_infos = textworld.EnvInfos(game=True, facts=True, feedback=True, inventory=True, location=True,
                                   last_action=True, last_command=True, intermediate_reward=True)
    env = textworld.start(args.game, wrappers=agent.wrappers, infos=env_infos)
    print("tw-oracle env=", env)
    agent.reset(env)
    if args.viewer is not None:
        from textworld.envs.wrappers import HtmlViewer
        env = HtmlViewer(env, port=args.viewer)

    if args.mode == "human" or args.very_verbose:
        print("Using {}.\n".format(env.__class__.__name__))

    if args.mode == "human" or args.verbose:
        env.render()

    # # game_state = env.reset()
    if hasattr(agent, 'get_initial_state'):
        game_state = agent.get_initial_state()   # retrieve initial game_state (from agent.reset(env))
    else:
        # game_state, _reward_, _done_ = env.step("look")
        if args.mode == 'oracle' or args.mode == 'asp':
            assert False
        game_state = env.reset()

    reward = 0
    done = False

    for _ in range(args.max_steps) if args.max_steps > 0 else itertools.count():
        command = agent.act(game_state, reward, done)
        game_state, reward, done = env.step(command)

        if args.mode == "human" or args.verbose:
            env.render()

        if done:
            break

    env.close()
    print("Done after {} steps. Score {}/{}.".format(game_state.moves, game_state.score, game_state.max_score))


if __name__ == "__main__":
    main()
