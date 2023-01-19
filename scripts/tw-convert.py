#!/usr/bin/env python

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.


import argparse
import inspect
import json
import re
from pathlib import Path, PurePath
from typing import Optional, Any, Tuple

from textworld.challenges import CHALLENGES
from textworld import GameMaker
from textworld.generator import Game, KnowledgeBase, GameOptions, compile_game, make_grammar
from textworld.generator.inform7 import Inform7Game
from twutils.tw_asp import generate_ASP_for_game

def rebuild_game(game, options:Optional[GameOptions], verbose=False):
 
    # Load knowledge base specific to this challenge.
    settings = game.metadata.get('settings', None)
    # print(settings)
    if not settings:
        print("WARNING: UNKNOWN SETTINGS")
        # print(game.metadata)
        if 'uuid' in game.metadata and '+drop' in game.metadata['uuid']:
            if verbose:
                print(f"DROP : {game.metadata['uuid']}")
        else:
            if verbose:
               print(f"NO DROP : {game.metadata['uuid']}")
    elif settings.get("drop"):
        if verbose:
            print("DROP")
        # options.kb = KnowledgeBase.load(logic_path=KB_LOGIC_DROP_PATH, grammar_path=KB_GRAMMAR_PATH)
    else:
        if verbose:
            print("NO DROP")
        # options.kb = KnowledgeBase.load(logic_path=KB_LOGIC_PATH, grammar_path=KB_GRAMMAR_PATH)

    assert options, f"MISSING REQUIRED options: {options}"
    assert options.path, f"MISSING REQUIRED options.path: {options.path}"
        # assert options.path, f"{options}"
    grammar = make_grammar(options.grammar, rng=options.rngs['grammar'], kb=options.kb)
    # print(options.grammar)
    for inf_id in game.infos:
        # print(inf_id, game.infos[inf_id].desc)
        if "cookbook" != game.infos[inf_id].name:
            game.infos[inf_id].desc = None  # wipe the full text descriptions to cause them to get regenerated
    # print("======================================================")
    game.change_grammar(grammar)
    # for inf_id in game.infos:
    #     print(inf_id, game.infos[inf_id].desc)
    return game


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert games to use a different grammar, or to ASP code")
    parser.add_argument("games", metavar="game", nargs="+",
                        help="JSON files containing infos about a game.")
    parser.add_argument("-c", "--challenge", choices=CHALLENGES, default=None,
                                help="Selects grammar for regenerating text descriptions")
    parser.add_argument("-s", "--simplify-grammar", action="store_true",
                        help="Regenerate tw-cooking games with simpler text grammar.")

    parser.add_argument("-o", "--output", default="./tw_games2/", metavar="PATH",
                                help="Path for the regenerated games. It should point to a folder,"
                                    " output filenames will be derived from the input filenames.")
    parser.add_argument("-f", "--format", choices=["ulx", "z8", "json"], default="z8",
                                help="Output format to use when recompiling games with --simplify-grammar. Default: %(default)s")

    parser.add_argument("-a", "--asp", action="store_true", help="Output ASP logic program.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose mode.")

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

        if args.simplify_grammar:
            output_file = (outpath / PurePath(input_filename).name).with_suffix("."+args.format)  #filename without path or suffix
            if args.verbose:
                print("OUTPUT SIMPLIFIED GAME:", output_file)
            options:GameOptions = GameOptions()
            # options.seeds = args.seed
            # dirname, basename = os.path.split(args.output)
            options.path = output_file
            options.file_ext = "." + args.format
            options.force_recompile = True

            options.grammar.theme = "simpler"
            options.kb = game.kb
            if args.challenge:
                challenge_file_path = Path(inspect.getfile(CHALLENGES[args.challenge][1])).parent
                # print(challenge_file_path)
                options.kb.text_grammars_path = f"{challenge_file_path}/textworld_data/text_grammars"
 
            # options.grammar.include_adj = args.include_adj
            # options.grammar.only_last_action = args.only_last_action
            # options.grammar.blend_instructions = args.blend_instructions
            # options.grammar.blend_descriptions = args.blend_descriptions
            # options.grammar.ambiguous_instructions = args.ambiguous_instructions
            # options.grammar.allowed_variables_numbering = args.entity_numbering


            objective0 = game.objective
            game = rebuild_game(game, options, verbose=args.verbose)
            game.objective = objective0
            if args.format == 'json':
                game.save(output_file) # output game as .json file
            else:
                game_file = compile_game(game, options)

            if args.verbose:
                print(f"--- NEW GAME KB:\n{game.kb}\n ---") # keys() = version, world, grammar, quests, infos, KB, metadata, objective 

        output_file = (outpath / PurePath(input_filename).name).with_suffix(".facts")  #filename without path or suffix
        if args.verbose:
            print(f"OUTPUT {'SIMPLIFIED ' if args.simplify_grammar else ''}FACTS: {output_file}")
        with open(output_file, "w") as outfile:
            _inform7 = Inform7Game(game)
            hfacts = list(map(_inform7.get_human_readable_fact, game.world.facts))
            #  game.kb.logic.serialize()
            json_out = {
                "rules": sorted([str(rule) for rule in game.kb.logic.rules.values()]),
                "types": game.kb.types.serialize(),
                "infos": [info.serialize() for info in game._infos.values()],
                "quests": [json.dumps(quest.serialize()) for quest in game.quests],
                "facts": [str(fact) for fact in game.world.facts],
                "hfacts": [str(fact) for fact in hfacts]
            }
            outfile.write(json.dumps(json_out, indent=2))
        if args.asp:
            asp_file = output_file.with_suffix(".lp")
            if args.verbose:
                print(f"ASP out: {asp_file}")
            generate_ASP_for_game(game, asp_file, hfacts=hfacts)

        # Path(destdir).mkdir(parents=True, exist_ok=True)
        # Path(make_dsfilepath(destdir, args.which)).unlink(missing_ok=True)


