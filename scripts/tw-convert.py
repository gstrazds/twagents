#!/usr/bin/env python

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.


import argparse
import inspect
import json
import re
from pathlib import Path, PurePath
from typing import Optional, Any

from textworld.challenges import CHALLENGES
from textworld import GameMaker
from textworld.logic import Proposition, Variable
from textworld.generator import Game, KnowledgeBase, GameOptions, compile_game, make_grammar
from textworld.generator.vtypes import VariableType, VariableTypeTree
from textworld.generator.inform7 import Inform7Game

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


def objname_to_asp(name:str) -> str:
    if name == 'P':
        aspname = 'player'
    elif name == 'I':
        aspname = 'inventory'
    else:
        aspname = name.lower()
    return aspname


#FLUENT_FACTS = ['at', 'on', 'in', 'open', 'closed', 'locked',
#  'uncut', 'raw', 'roasted', 'edible', 'inedible', 'sliced', 'diced', 'chopped']
STATIC_FACTS = ['cuttable', 'cookable', 'edible', 'link', 'north_of', 'east_of', 'south_of', 'west_of']

def fact_to_asp(fact:Proposition, hfact=None, step=0) -> str:
    """ converts a TextWorld fact to a format appropriate for Answer Set Programming"""
    asp_args = [Variable(objname_to_asp(v.name), v.type) for v in fact.arguments]
    asp_fact = Proposition(fact.name, asp_args)
    asp_fact_str = re.sub(r":[^,)]*", '', f"{str(asp_fact)}.")
    if fact.name not in STATIC_FACTS \
        and 'ingredient' not in asp_fact_str:   #TextWorld quirk: facts refering to 'ingredient_N' are static (define the recipe)
        asp_fact_str = asp_fact_str.replace(").", f", {step}).")  # add the time step

    if hfact:
        asp_fact_str = f"{asp_fact_str} % {str(hfact)}"
    return asp_fact_str


def twtype_to_typename(type:str) -> str:
    return 'thing' if type == 't' else objname_to_asp(type)


eg_types_to_asp = """
"types": [
    "I: I",
    "P: P",
    "RECIPE: RECIPE",
    "c: c -> t",
    "d: d -> t",
    "f: f -> o",
    "ingredient: ingredient -> t",
    "k: k -> o",
    "meal: meal -> f",
    "o: o -> t",
    "object: object",
    "oven: oven -> c",
    "r: r",
    "s: s -> t",
    "slot: slot",
    "stove: stove -> s",
    "t: t",
    "toaster: toaster -> c"
  ],
 """

INC_MODE = \
"""#include <incmode>.
#const imax=500.  % give up after this many iterations
%#script (python)
%
%from clingo import Function, Symbol, Number
%
%def get(val, default):
%    return val if val != None else default
%
%def main(prg):
%    imin = get(prg.get_const("imin"), 1)
%    imax = get(prg.get_const("imax"), 500)
%    istop = get(prg.get_const("istop"), "SAT")
%
%    step, ret = 0, None
%    while ((imax is None or step < imax) and
%           (step == 0 or step <= imin or (
%              (istop == "SAT" and not ret.satisfiable) or
%              (istop == "UNSAT" and not ret.unsatisfiable) or
%              (istop == "UNKNOWN" and not ret.unknown))
%           )):
%        parts = []
%        parts.append(("check", [Number(step)]))
%        if step > 0:
%            prg.release_external(Function("query", [Number(step-1)]))
%            parts.append(("step", [Number(step)]))
%            prg.cleanup()
%        else:
%            parts.append(("base", []))
%        prg.ground(parts)
%        prg.assign_external(Function("query", [Number(step)]), True)
%        ret, step = prg.solve(), step+1
%#end.
%
%#program check(t).
%#external query(t).

#program base.
% Define
"""


TYPE_RULES = \
"""
subclass_of(A,C) :- subclass_of(A,B), subclass_of(B,C).
instance_of(I,B) :- subclass_of(A,B), instance_of(I,A).
direction(east). direction(west). direction(north). direction(south).

"""
#  "class(A) :- instance_of(I,A).

MAP_RULES = \
"""
connected(R1,R2,west) :- west_of(R1, R2), r(R1), r(R2).
connected(R1,R2,north) :- north_of(R1, R2), r(R1), r(R2).
connected(R1,R2,south) :- south_of(R1, R2), r(R1), r(R2).
connected(R1,R2,east) :- east_of(R1, R2), r(R1), r(R2).

% all doors/exits can be traversed in both directions
connected(R1,R2,east) :- connected(R2,R1,west).
connected(R1,R2,west) :- connected(R2,R1,east).
connected(R1,R2,south) :- connected(R2,R1,north).
connected(R1,R2,north) :- connected(R2,R1,south).

connected(R1,R2) :- connected(R1,R2,_).

atP(0,R) :- at(player,R,0), r(R).   % Alias for player's initial position
"""


NAV_RULES = \
"""
#program step(t).
% Generate
timestep(t).
0 {at(player,R,t):r(R)} 1 :- at(player,R0,t-1), connected(R0,R), r(R0), r(R). %, T<=maxT. % can move to an adjacent room 
0 {at(player,R0,t):r(R0)} 1 :- at(player,R0,t-1), r(R0). %, T<=maxT.  % can stay in the current room
1 {at(player,R,t):r(R)} 1 :- timestep(t).   % player is in exactly one room at any given time
% Define
movedP(t,R0,R) :- at(player,R0,t-1), r(R0), r(R), at(player,R,t), R!=R0.   % alias for player moved at time t
atP(t,R) :- at(player,R,t).                                       % Alias for current room
% Test
:- at(player,R0,t-1), at(player,R,t), r(R0), r(R), R!=R0, not connected(R,R0).

"""



GAME_RULES_OLD = \
"""
% ------ MAKE ------
% make/recipe/1 :: $at(P, r) & $cooking_location(r, RECIPE) & in(f, I) & $ingredient_1(f) & $out(meal, RECIPE) -> in(meal, I) & edible(meal) & used(f) & raw(meal)
% make/recipe/2 :: $at(P, r) & $cooking_location(r, RECIPE) & in(f, I) & $ingredient_1(f) & in(f', I) & $ingredient_2(f') & $out(meal, RECIPE) -> in(meal, I) & edible(meal) & used(f) & used(f') & raw(meal)
% make/recipe/3 :: $at(P, r) & $cooking_location(r, RECIPE) & in(f, I) & $ingredient_1(f) & in(f', I) & $ingredient_2(f') & in(f'', I) & $ingredient_3(f'') & $out(meal, RECIPE) -> in(meal, I) & edible(meal) & used(f) & used(f') & used(f'') & raw(meal)
% make/recipe/4 :: $at(P, r) & $cooking_location(r, RECIPE) & in(f, I) & $ingredient_1(f) & in(f', I) & $ingredient_2(f') & in(f'', I) & $ingredient_3(f'') & in(f''', I) & $ingredient_4(f''') & $out(meal, RECIPE) -> in(meal, I) & edible(meal) & used(f) & used(f') & used(f'') & used(f''') & raw(meal)
% make/recipe/5 :: $at(P, r) & $cooking_location(r, RECIPE) & in(f, I) & $ingredient_1(f) & in(f', I) & $ingredient_2(f') & in(f'', I) & $ingredient_3(f'') & in(f''', I) & $ingredient_4(f''') & in(f'''', I) & $ingredient_5(f'''') & $out(meal, RECIPE) -> in(meal, I) & edible(meal) & used(f) & used(f') & used(f'') & used(f''') & used(f'''') & raw(meal)


% ------ CONSUME ------
% drink :: in(f, I) & drinkable(f) -> consumed(f)
% eat :: in(f, I) & edible(f) -> consumed(f)

% ------ TAKE ------
% take :: $at(P, r) & at(o, r) -> in(o, I)
% take/c :: $at(P, r) & $at(c, r) & $open(c) & in(o, c) -> in(o, I)
% take/s :: $at(P, r) & $at(s, r) & on(o, s) -> in(o, I)

% ------ DROP/PUT ------
% put :: $at(P, r) & $at(s, r) & in(o, I) -> on(o, s)
% drop :: $at(P, r) & in(o, I) -> at(o, r)
% insert :: $at(P, r) & $at(c, r) & $open(c) & in(o, I) -> in(o, c)

"""


GAME_RULES_COMMON = \
"""
% ------ LOOK ------
% examine/I :: $at(o, I) -> 
% examine/c :: $at(P, r) & $at(c, r) & $open(c) & $in(o, c) -> 
% examine/s :: $at(P, r) & $at(s, r) & $on(o, s) -> 
% examine/t :: $at(P, r) & $at(t, r) -> 
% look :: $at(P, r) -> 
% inventory :: $at(P, r) -> 

% ------ GO ------
% go/east :: at(P, r) & $west_of(r, r') & $free(r, r') & $free(r', r) -> at(P, r')
% go/north :: at(P, r) & $north_of(r', r) & $free(r, r') & $free(r', r) -> at(P, r')
% go/south :: at(P, r) & $north_of(r, r') & $free(r, r') & $free(r', r) -> at(P, r')
% go/west :: at(P, r) & $west_of(r', r) & $free(r, r') & $free(r', r) -> at(P, r')


% ------ OPEN/CLOSE UNLOCK/LOCK ------
% close/c :: $at(P, r) & $at(c, r) & open(c) -> closed(c)
% close/d :: $at(P, r) & $link(r, d, r') & $link(r', d, r) & open(d) & free(r, r') & free(r', r) -> closed(d)
% open/c :: $at(P, r) & $at(c, r) & closed(c) -> open(c)
% open/d :: $at(P, r) & $link(r, d, r') & $link(r', d, r) & closed(d) -> open(d) & free(r, r') & free(r', r)
% lock/c :: $at(P, r) & $at(c, r) & $in(k, I) & $match(k, c) & closed(c) -> locked(c)
% lock/d :: $at(P, r) & $link(r, d, r') & $link(r', d, r) & $in(k, I) & $match(k, d) & closed(d) -> locked(d)
% unlock/c :: $at(P, r) & $at(c, r) & $in(k, I) & $match(k, c) & locked(c) -> closed(c)
% unlock/d :: $at(P, r) & $link(r, d, r') & $link(r', d, r) & $in(k, I) & $match(k, d) & locked(d) -> closed(d)

% ------ COOK ------
% cook/oven/burned :: $at(P, r) & $at(oven, r) & $in(f, I) & cooked(f) & edible(f) -> burned(f) & inedible(f)
% cook/oven/cooked/needs_cooking :: $at(P, r) & $at(oven, r) & $in(f, I) & needs_cooking(f) & inedible(f) -> roasted(f) & edible(f) & cooked(f)
% cook/oven/cooked/raw :: $at(P, r) & $at(oven, r) & $in(f, I) & raw(f) -> roasted(f) & cooked(f)
% cook/stove/burned :: $at(P, r) & $at(stove, r) & $in(f, I) & cooked(f) & edible(f) -> burned(f) & inedible(f)
% cook/stove/cooked/needs_cooking :: $at(P, r) & $at(stove, r) & $in(f, I) & needs_cooking(f) & inedible(f) -> fried(f) & edible(f) & cooked(f)
% cook/stove/cooked/raw :: $at(P, r) & $at(stove, r) & $in(f, I) & raw(f) -> fried(f) & cooked(f)
% cook/toaster/burned :: $at(P, r) & $at(toaster, r) & $in(f, I) & cooked(f) & edible(f) -> burned(f) & inedible(f)
% cook/toaster/cooked/needs_cooking :: $at(P, r) & $at(toaster, r) & $in(f, I) & needs_cooking(f) & inedible(f) -> grilled(f) & edible(f) & cooked(f)
% cook/toaster/cooked/raw :: $at(P, r) & $at(toaster, r) & $in(f, I) & raw(f) -> grilled(f) & cooked(f)

% ------ CUT ------
% chop :: $in(f, I) & $in(o, I) & $sharp(o) & uncut(f) -> chopped(f)
% dice :: $in(f, I) & $in(o, I) & $sharp(o) & uncut(f) -> diced(f)
% slice :: $in(f, I) & $in(o, I) & $sharp(o) & uncut(f) -> sliced(f)


% ------ CONSTRAINTS ------
:- open(X,t); closed(X,t).    % any door or container can be either open or closed but not both
:- locked(X,t), open(X,t).    % can't be both locked and open at the same time

"""

GAME_RULES_NEW = \
"""
% ------ MAKE ------
%- make/recipe/1 :: $at(P, r) & $cooking_location(r, RECIPE) & in(f, I) & $ingredient_1(f) & $out(meal, RECIPE) -> in(meal, I) & edible(meal) & used(f) & raw(meal)
%+ make/recipe/1 :: $at(P, r) & $cooking_location(r, RECIPE) & in(f, I) & $ingredient_1(f) & $out(meal, RECIPE) -> in(meal, I) & edible(meal) & used(f) & raw(meal)",
%- make/recipe/2 :: $at(P, r) & $cooking_location(r, RECIPE) & in(f, I) & $ingredient_1(f) & in(f', I) & $ingredient_2(f') & $out(meal, RECIPE) -> in(meal, I) & edible(meal) & used(f) & used(f') & raw(meal)
%+  make/recipe/2 :: $at(P, r) & $cooking_location(r, RECIPE) & in(f, I) & $ingredient_1(f) & in(f', I) & $ingredient_2(f') & $out(meal, RECIPE) & $used(slot) & used(slot') -> in(meal, I) & free(slot') & edible(meal) & used(f) & used(f') & raw(meal)",
%- make/recipe/3 :: $at(P, r) & $cooking_location(r, RECIPE) & in(f, I) & $ingredient_1(f) & in(f', I) & $ingredient_2(f') & in(f'', I) & $ingredient_3(f'') & $out(meal, RECIPE) -> in(meal, I) & edible(meal) & used(f) & used(f') & used(f'') & raw(meal)
%+ make/recipe/3 :: $at(P, r) & $cooking_location(r, RECIPE) & in(f, I) & $ingredient_1(f) & in(f', I) & $ingredient_2(f') & in(f'', I) & $ingredient_3(f'') & $out(meal, RECIPE) & $used(slot) & used(slot') & used(slot'') -> in(meal, I) & free(slot') & free(slot'') & edible(meal) & used(f) & used(f') & used(f'') & raw(meal)",
%- make/recipe/4 :: $at(P, r) & $cooking_location(r, RECIPE) & in(f, I) & $ingredient_1(f) & in(f', I) & $ingredient_2(f') & in(f'', I) & $ingredient_3(f'') & in(f''', I) & $ingredient_4(f''') & $out(meal, RECIPE) -> in(meal, I) & edible(meal) & used(f) & used(f') & used(f'') & used(f''') & raw(meal)
%+ make/recipe/4 :: $at(P, r) & $cooking_location(r, RECIPE) & in(f, I) & $ingredient_1(f) & in(f', I) & $ingredient_2(f') & in(f'', I) & $ingredient_3(f'') & in(f''', I) & $ingredient_4(f''') & $out(meal, RECIPE) & $used(slot) & used(slot') & used(slot'') & used(slot'') -> in(meal, I) & free(slot') & free(slot'') & free(slot''') & edible(meal) & used(f) & used(f') & used(f'') & used(f''') & raw(meal)",
%- make/recipe/5 :: $at(P, r) & $cooking_location(r, RECIPE) & in(f, I) & $ingredient_1(f) & in(f', I) & $ingredient_2(f') & in(f'', I) & $ingredient_3(f'') & in(f''', I) & $ingredient_4(f''') & in(f'''', I) & $ingredient_5(f'''') & $out(meal, RECIPE) -> in(meal, I) & edible(meal) & used(f) & used(f') & used(f'') & used(f''') & used(f'''') & raw(meal)
%+ make/recipe/5 :: $at(P, r) & $cooking_location(r, RECIPE) & in(f, I) & $ingredient_1(f) & in(f', I) & $ingredient_2(f') & in(f'', I) & $ingredient_3(f'') & in(f''', I) & $ingredient_4(f''') & in(f'''', I) & $ingredient_5(f'''') & $out(meal, RECIPE) & $used(slot) & used(slot') & used(slot'') & used(slot''') & used(slot'''') -> in(meal, I) & free(slot') & free(slot'') & free(slot''') & free(slot'''') & edible(meal) & used(f) & used(f') & used(f'') & used(f''') & used(f'''') & raw(meal)",


% ------ CONSUME ------
% drink :: in(f, I) & drinkable(f) -> consumed(f)
%+ drink :: in(f, I) & drinkable(f) & used(slot) -> consumed(f) & free(slot)",
% eat :: in(f, I) & edible(f) -> consumed(f)
%+ eat :: in(f, I) & edible(f) & used(slot) -> consumed(f) & free(slot)",


% ------ TAKE ------
%- take :: $at(P, r) & at(o, r) -> in(o, I)
%+ take :: $at(P, r) & at(o, r) & free(slot) -> in(o, I) & used(slot)",
%- take/c :: $at(P, r) & $at(c, r) & $open(c) & in(o, c) -> in(o, I)
%+ take/c :: $at(P, r) & $at(c, r) & $open(c) & in(o, c) & free(slot) -> in(o, I) & used(slot)",
%- take/s :: $at(P, r) & $at(s, r) & on(o, s) -> in(o, I)
%+ take/s :: $at(P, r) & $at(s, r) & on(o, s) & free(slot) -> in(o, I) & used(slot)",

% ------ DROP/PUT ------
% put :: $at(P, r) & $at(s, r) & in(o, I) -> on(o, s)
%+ put :: $at(P, r) & $at(s, r) & in(o, I) & used(slot) -> on(o, s) & free(slot)",
% drop :: $at(P, r) & in(o, I) -> at(o, r)
%+ drop :: $at(P, r) & in(o, I) & used(slot) -> at(o, r) & free(slot)",
% insert :: $at(P, r) & $at(c, r) & $open(c) & in(o, I) -> in(o, c)
%+ insert :: $at(P, r) & $at(c, r) & $open(c) & in(o, I) & used(slot) -> in(o, c) & free(slot)",

--------------------------------------------------------------------------------


"""

def types_to_asp(typestree: VariableTypeTree) -> str:
    # typestree.serialize(): return [vtype.serialize() for vtype in self.variables_types.values()]
    def _vtype_info(vtype:Variable) -> str:
        info_tuple = (twtype_to_typename(vtype.name), twtype_to_typename(vtype.parent) if vtype.parent else None)
        return info_tuple
    type_infos = [_vtype_info(vtype) for vtype in typestree.variables_types.values()]
    return type_infos


def info_to_asp(info) -> str:
    type_name = twtype_to_typename(info.type)
    info_type_str = f"{type_name}({objname_to_asp(info.id)})."
    if info.name:
        info_type_str += f" % {(info.adj if info.adj else '')} {info.name}"
    return info_type_str


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
            with open(asp_file, "w") as aspfile:
                aspfile.write(INC_MODE)
                aspfile.write("% ------- Types -------\n")
                aspfile.write(TYPE_RULES)
                type_infos = types_to_asp(game.kb.types)
                for typename, _ in type_infos:
                    # aspfile.write(f"class({typename}). ") . # can derive this automatically from instance_of() or subclass_of()
                    aspfile.write(f"instance_of(X,{typename}) :- {typename}(X).\n")
                for typename, parent_type in type_infos:
                    if parent_type:  
                        # and parent_type != 'thing':  # currently have no practical use for 'thing' base class (dsintinugishes objects from rooms)
                        aspfile.write(f"subclass_of({typename},{parent_type}).\n")
                aspfile.write("\n% ------- Things -------\n")
                for info in game._infos.values():
                    aspfile.write(info_to_asp(info))
                    aspfile.write('\n')
            #  game.kb.logic.serialize()
                aspfile.write("\n% ------- Facts -------\n")
                for fact, hfact in zip(game.world.facts, hfacts):
                    aspfile.write(fact_to_asp(fact, hfact, step=0))
                    aspfile.write('\n')
                aspfile.write("\n% ------- Navigation -------\n")
                aspfile.write(MAP_RULES)
                aspfile.write(NAV_RULES)
                # ---- GAME DYNAMICS
                aspfile.write(GAME_RULES_COMMON)

                aspfile.write(
                    "#program check(t).\n" \
                    "% Test\n" \
                    ":- at(player,R,t), r(R), R != goalR, query(t) . % Fail if we don't end up in the target room [kitchen =r_0]\n"
                )
                #aspfile.write(":- movedP(T,R,R1), at(player,R1,T0), timestep(T0), T0<T .  % disallow loops\n")
                # For branch & bound optimization:
                # aspfile.write( #":- not at(player,r_0,maxT).  % end up in the kitchen\n")
                #     "ngoal(T) :- at(player,R,T), r(R), R!=r_0 .  % want to end up in the kitchen (r_0)\n" \
                #     ":- ngoal(maxT).\n  % anti-goal -- fail if goal not achieved"
                # )
                #aspfile.write("_minimize(1,T) :- ngoal(T).\n")

                aspfile.write("#show movedP/3.\n")
                aspfile.write("#show atP/2.\n")

        # Path(destdir).mkdir(parents=True, exist_ok=True)
        # Path(make_dsfilepath(destdir, args.which)).unlink(missing_ok=True)


