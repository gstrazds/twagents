#!/usr/bin/env python

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.


import sys
import argparse
import json
from pathlib import Path, PurePath
from typing import Optional, Any

from textworld.generator import Game, KnowledgeBase, GameOptions
from textworld import GameMaker
from textworld.generator.inform7 import Inform7Game

def rebuild_game(game, options:Optional[GameOptions]):
 
    # Load knowledge base specific to this challenge.
    settings = game.metadata.get('settings', None)
    print(settings)
    if not settings:
        print("UNKNOWN SETTINGS")
        # print(game.metadata)
        if 'uuid' in game.metadata and '+drop+' in game.metadata['uuid']:
            print(f"DROP : {game.metadata['uuid']}")
        else:
            print(f"NO DROP : {game.metadata['uuid']}")
    elif settings.get("drop"):
        print("DROP")
        # options.kb = KnowledgeBase.load(logic_path=KB_LOGIC_DROP_PATH, grammar_path=KB_GRAMMAR_PATH)
    else:
        print("NO DROP")
        # options.kb = KnowledgeBase.load(logic_path=KB_LOGIC_PATH, grammar_path=KB_GRAMMAR_PATH)

    if not options or not options.path:
        print(f"MISSING options.path: {options}")
        # assert options.path, f"{options}"
    return game


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert games to use a different grammar, or to ASP code")
    parser.add_argument("games", metavar="game", nargs="+",
                        help="JSON files containing infos about a game.")
    parser.add_argument("--simplify-grammar", action="store_true",
                        help="Regenerate tw-cooking games with simpler text grammar.")

    parser.add_argument("--output", default="./tw_games2/", metavar="PATH",
                                help="Path for the regenerated games. It should point to a folder,"
                                    " output filenames will be derived from the input filenames.")
    parser.add_argument('--format', choices=["ulx", "z8", "json"], default="z8",
                                help="Which format to use when compiling the game. Default: %(default)s")

    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose mode.")
    args = parser.parse_args()
    outpath = Path(args.output)
    if outpath.exists():
        assert outpath.is_dir(), f"Output path >>{args.output}<< should be a directory!"
    else:
        outpath.mkdir(mode=0o775, parents=True)

    objectives = {}
    names = set()
    for game_filename in args.games:
        try:
            input_filename = game_filename.replace(".ulx", ".json")
            input_filename = input_filename.replace(".z8", ".json")
            game = Game.load(input_filename)
        except Exception as e:
            print("**** Failed: Game.load({}).".format(input_filename))
            print(e)
            continue
        output_file = (outpath / PurePath(input_filename).name).with_suffix(".facts")  #filename without path or suffix
        print("FACTS OUT:", output_file)
        with open(output_file, "w") as outfile:
            _inform7 = Inform7Game(game)
            hfacts = list(map(_inform7.get_human_readable_fact, game.world.facts))
            #  game.kb.logic.serialize()
            json_out = {
                "rules": sorted([str(rule) for rule in game.kb.logic.rules.values()]),
                "types": game.kb.types.serialize(),
                "infos": [info.serialize() for info in game._infos.values()],
                "facts": [str(fact) for fact in game.world.facts],
                "hfacts": [str(fact) for fact in hfacts]
            }
            outfile.write(json.dumps(json_out, indent=2))
        if args.simplify_grammar:
            output_file = (outpath / PurePath(input_filename).name).with_suffix("."+args.format)  #filename without path or suffix
            print("OUTPUT SIMPLIFIED GAME:", output_file)
            options:GameOptions = None
            new_game = rebuild_game(game, options)
            print(f"NEW GAME (KB): {new_game.kb}") # keys() = version, world, grammar, quests, infos, KB, metadata, objective 
            print(f"NEW GAME (grammar): {new_game.grammar}") # keys() = version, world, grammar, quests, infos, KB, metadata, objective 
        if args.verbose:
            print("ORIG GAME:", game.serialize())

        # Path(destdir).mkdir(parents=True, exist_ok=True)
        # Path(make_dsfilepath(destdir, args.which)).unlink(missing_ok=True)


