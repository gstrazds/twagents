#!/usr/bin/env python

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.


import argparse
import inspect
import json
import re
from pathlib import Path, PurePath
from typing import Optional, Any, Tuple

import gymnasium as gym


MG_SPECS = [
#    'MiniGrid-FourRooms-v0',

]

if __name__ == "__main__":
    import pathlib
    import sys
    parent_dir = pathlib.Path(__file__).resolve().parents[1]
    sys.path.append(str(parent_dir))

    from mgutils.mg_asp import generate_ASP_for_minigrid

    for spec in gym.registry.values():
        #namespace, base_id, ver = gym.envs.registration.parse_env_id(spec.id)
        if spec.id.startswith("MiniGrid") or spec.id.startswith("BabyAI"):
            MG_SPECS.append(spec.id)

    parser = argparse.ArgumentParser(description="Convert minigrid games to ASP")
    parser.add_argument("-s", "--spec", default='MiniGrid-FourRooms-v0',  #choices=MG_SPECS, 
                                help="minigrid or babyAI spec")
    parser.add_argument("--seed", default=None,  #choices=MG_SPECS, 
                                help="RNG seed for initial game config")
    parser.add_argument("-o", "--output", default="./mg_games/", metavar="PATH",
                                help="Path for the converted games. It should point to a folder,"
                                    " output filename will be derived from the input spec.")

    parser.add_argument("--no-python", action="store_true", help="Don't embed the python solver loop")
    parser.add_argument("--standalone", action="store_true", help="Include shared ASP rules for TW game dynamics")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose mode.")
    parser.add_argument("--do-write", action="store_true", help="Save .lp file to output dir")
    args = parser.parse_args()
    if not args.spec or args.spec not in MG_SPECS:
        print("Please choose one of the following registered minigrid configs:")
        for spec_id in MG_SPECS:
            print('\t', spec_id)
            exit()
    else:
        print("MG CONVERT: spec id =", args.spec)

    outpath = Path(args.output)
    if outpath.exists():
        assert outpath.is_dir(), f"Output path >>{args.output}<< should be a directory not a file!"
    else:
        if args.do_write:
            outpath.mkdir(mode=0o775, parents=True)

    output_file = (outpath / args.spec).with_suffix(".lp")

    if args.verbose:
        print(f"ASP out: {output_file}" +
          ' (standalone)' if args.standalone else '' +
          ' (no Python)' if args.no_python else '')
    if not args.no_python:
        emb_python = True
    else:
        emb_python = False
    
    env = gym.make(args.env)  #, tile_size=args.tile_size, max_steps=args.max_steps)
    # if args.xplore_bonus:
    #     env = ActionBonus(env)
    # env = AccumScore(env)
    # print(f"{env.unwrapped.spec}")
    # min_reward, max_reward = map(float, env.reward_range)
    # if args.agent_view:
    #     print("Using agent view")
    #     env = RGBImgPartialObsWrapper(env, env.tile_size)
    #     env = ImgObsWrapper(env)

    asp_str = generate_ASP_for_minigrid(env,
        seed=args.seed,
        standalone=args.standalone,
        emb_python=emb_python)

    if args.verbose and args.do_write:
        print(f"WRITING OUTPUT FILE: {output_file}")
        with open(output_file, "w") as outfile:
            outfile.write(asp_str)


