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


#FLUENT_FACTS = ['at', 'on', 'in',
#  'open', 'closed', 'locked',
#  'uncut', , 'sliced', 'diced', 'chopped',
#  'raw', 'roasted', 'grilled', 'fried', 'edible', 'inedible']
STATIC_FACTS = ['cuttable', 'cookable', 'sharp', 'link', 'cooking_location', 'north_of', 'east_of', 'south_of', 'west_of']

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
%from clingo import Function, Symbol, String, Number
%
%def get(val, default):
%    return val if val != None else default
%
%def main(prg):
%    imin = get(prg.get_const("imin"), 1)
%    imax = get(prg.get_const("imax"), 500)
%    istop = get(prg.get_const("istop"), String("SAT"))
%
%    step, ret = 0, None
%    while ((imax is None or step < imax.number) and
%           (step == 0 or step <= imin.number or (
%              (istop == "SAT" and not ret.satisfiable) or
%              (istop == "UNSAT" and not ret.unsatisfiable) or
%              (istop == "UNKNOWN" and not ret.unknown))
%           )):
%        parts = []
%        parts.append(("check", [Number(step)]))
%        if step > 0:
%            prg.release_external(Function("query", [Number(step-1)]))
%            parts.append(("step", [Number(step)]))
%            %? prg.cleanup()
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
subclass_of(A,C) :- subclass_of(A,B), subclass_of(B,C).  % subclass relationship is transitive
instance_of(I,B) :- instance_of(I,A), subclass_of(A,B).  % an instance of a subclass is also an instance of the superclass
class(C) :- instance_of(X,C).  % instance_of relationship implicity defines classes
class(S) :- subclass_of(S,C).  % every subclass is also a class
class(S) :- subclass_of(S,C).  % redundant [with instance-of-superclass rule, above]

% additional inheritance links to consolidate 3 rules for cooking with appliances
class(cooker).
subclass_of(oven,cooker).
subclass_of(stove,cooker).
subclass_of(toaster,cooker).

{is_openable(X); is_lockable(X)}=2 :- instance_of(X,d). % doors are potentially openable and lockable
is_openable(X) :- instance_of(X,c).  % containers are potentially openable
is_lockable(X) :- instance_of(X,c), not instance_of(X,oven). % most containers are potentially lockable, but ovens are not

% action vocabulary
timestep(0). % incremental solving will define timestep(t) for t >= 1...
direction(east;west;north;south).
arg(none).   % placeholder for actions with fewer than 2 args
arg(NSEW) :- direction(NSEW).
arg(I) :- instance_of(I,C), class(C).

cutting_verb(chop;slice;dice).

"""
#  "class(A) :- instance_of(I,A).

MAP_RULES = \
"""
connected(R1,R2,west) :- west_of(R1, R2), r(R1), r(R2).
connected(R1,R2,north) :- north_of(R1, R2), r(R1), r(R2).
connected(R1,R2,south) :- south_of(R1, R2), r(R1), r(R2).
connected(R1,R2,east) :- east_of(R1, R2), r(R1), r(R2).

% assume that all doors/exits can be traversed in both directions
connected(R1,R2,east) :- connected(R2,R1,west).
connected(R1,R2,west) :- connected(R2,R1,east).
connected(R1,R2,south) :- connected(R2,R1,north).
connected(R1,R2,north) :- connected(R2,R1,south).

connected(R1,R2) :- connected(R1,R2,NSEW), direction(NSEW).
has_door(R,D) :- r(R), d(D), link(R,D,_).
door_direction(R,D,NSEW) :- r(R), r(R2), d(D), direction(NSEW), link(R,D,R2), connected(R,R2,NSEW).

atP(0,R) :- at(player,R,0), r(R).   % Alias for player's initial position
"""


ACTION_STEP_RULES = \
"""

#program step(t).

% Generate
timestep(t).

{act(X,t):is_action(X,t)} = 1 :- timestep(t). % player must choose exactly one action at each time step.

{at(player,R,t):r(R)} = 1 :- timestep(t).   % player is in exactly one room at any given time

% NOTE/IMPORTANT - THE FOLLOWING IS NOT THE SAME as prev line, DOES NOT WORK CORRECTLY:
%  {at(player,R,t)} = 1 :- r(R), timestep(t).
% NOTE - THE FOLLOWING ALSO DOESN'T WORK (expands too many do_moveP ground instances at final timestep:
%  {at(player,R,t)} = 1 :- r(R), at(player,R,t), timestep(t).


% define fluents that determine whether player can move from room R0 to R1
free(R0,R1,t) :- r(R0), r(R1), d(D), link(R0,D,R1), open(D,t).
not free(R0,R1,t) :- r(R0), r(R1), d(D), link(R0,D,R1), not open(D,t). 
free(R0,R1,t) :- r(R0), r(R1), connected(R0,R1), not link(R0,_,R1).  % if there is no door, exit is always traversible.



% Alias
openD(D,t) :- open(D,t), d(D), timestep(t).  % alias for 'door D is open at time t'
atP(t,R) :- at(player,R,t).                  % alias for player's current location
in_inventory(O,t) :- in(O,inventory,t).      % alias for object is in inventory at time t

"""

GAME_RULES_COMMON = \
"""
% ------ LOOK ------
% inventory :: $at(P, r) -> 
% look :: $at(P, r) -> 

% ------ LOOK AT: EXAMINE an object ------
% examine/I :: $at(o, I) -> 
% examine/c :: $at(P, r) & $at(c, r) & $open(c) & $in(o, c) -> 
% examine/s :: $at(P, r) & $at(s, r) & $on(o, s) -> 
% examine/t :: $at(P, r) & $at(t, r) -> 

% examine/c :: can examine an object that's in a container if the container is open and player is in the same room
0 {do_examine(t,O)} 1 :- at(player,R,t), r(R), instance_of(O,o), instance_of(C,c), at(C,R,t), in(O,C,t), open(C,t), timestep(t).
% examine/s :: can examine an object that is on a support if player is in the same room as the suppport
0 {do_examine(t,O)} 1 :- at(player,R,t), r(R), instance_of(O,o), s(S), at(S,R,t), on(O,S,t), timestep(t).
% examine/t :: can examine a thing if player is in the same room as the thing
0 {do_examine(t,O)} 1 :- at(player,R,t), r(R), instance_of(O,thing), at(O,R,t), timestep(t).

% Test constraints
% have to be in the same room to examine something
:- do_examine(t,O), at(player,R,t), o(O), r(R), on(O,S,t), at(S,R2,t), s(S), r(R2), timestep(t), R != R2.
:- do_examine(t,O), at(player,R,t), o(O), r(R), in(O,C,t), at(C,R2,t), c(C), r(R2), timestep(t), R != R2.
:- do_examine(t,O), at(player,R,t), o(O), r(R), at(O,R2,t), r(R2), timestep(t), R != R2.

is_action(do_examine(t,O), t) :- do_examine(t,O). %, instance_of(O,thing).


% ------ GO ------
% go/east :: at(P, r) & $west_of(r, r') & $free(r, r') & $free(r', r) -> at(P, r')
% go/north :: at(P, r) & $north_of(r', r) & $free(r, r') & $free(r', r) -> at(P, r')
% go/south :: at(P, r) & $north_of(r, r') & $free(r, r') & $free(r', r) -> at(P, r')
% go/west :: at(P, r) & $west_of(r', r) & $free(r, r') & $free(r', r) -> at(P, r')

% --- inertia: player stays in the current room unless acts to move to another
  % stay in the current room unless current action is do_moveP
at(player,R0,t) :- at(player,R0,t-1), r(R0), {act(do_moveP(t,R0,R,NSEW),t):r(R),direction(NSEW)}=0. %, T<=maxT.

  % player moved at time t, from previous room R0 to new room R
at(player,R,t) :- act(do_moveP(t,R0,R,NSEW),t), at(player,R0,t-1), r(R0), r(R), connected(R0,R,NSEW), direction(NSEW). %, R!=R0.

% Test constraints
:- at(player,R0,t-1), at(player,R,t), r(R0), r(R), R!=R0, not free(R,R0,t-1).

 % can move to a connected room, if not blocked by a closed door
0 {do_moveP(t,R0,R,NSEW):free(R0,R,t-1),direction(NSEW)} 1 :- at(player,R0,t-1), connected(R0,R,NSEW), direction(NSEW), r(R0), r(R). %, T<=maxT.
% Test constraints
:- do_moveP(t,R0,R,NSEW),direction(NSEW),r(R0),r(R),timestep(t),not free(R0,R,t-1).  % can't go that way: not a valid action

is_action(do_moveP(t,R1,R2,NSEW), t) :- do_moveP(t,R1,R2,NSEW). %, r(R1), r(R2), direction(NSEW).


% ------ OPEN/CLOSE UNLOCK/LOCK ------
% close/c :: $at(P, r) & $at(c, r) & open(c) -> closed(c)
% close/d :: $at(P, r) & $link(r, d, r') & $link(r', d, r) & open(d) & free(r, r') & free(r', r) -> closed(d)
% open/c :: $at(P, r) & $at(c, r) & closed(c) -> open(c)
% open/d :: $at(P, r) & $link(r, d, r') & $link(r', d, r) & closed(d) -> open(d) & free(r, r') & free(r', r)
% lock/c :: $at(P, r) & $at(c, r) & $in(k, I) & $match(k, c) & closed(c) -> locked(c)
% lock/d :: $at(P, r) & $link(r, d, r') & $link(r', d, r) & $in(k, I) & $match(k, d) & closed(d) -> locked(d)
% unlock/c :: $at(P, r) & $at(c, r) & $in(k, I) & $match(k, c) & locked(c) -> closed(c)
% unlock/d :: $at(P, r) & $link(r, d, r') & $link(r', d, r) & $in(k, I) & $match(k, d) & locked(d) -> closed(d)

% --- inertia: doors and containers don't change state unless player acts on them
open(X,t) :- is_openable(X), open(X,t-1), not act(do_close(t,X)).
open(X,t) :- is_openable(X), act(do_open(t,X),t), not open(X,t-1).
closed(X,t) :- closed(X,t-1), not act(do_open(t,X),t).
locked(X,t) :- locked(X,t-1), not act(do_unlock(t,X,_),t).
% ------ CONSTRAINTS ------
:- open(X,t), closed(X,t).  %[,is_openable(X).]    % any door or container can be either open or closed but not both
:- locked(X,t), open(X,t).  %[,is_lockable(X).]    % can't be both locked and open at the same time

% can open a closed but unlocked door
0 {do_open(t,D)} 1 :- at(player,R0,t-1), r(R0), r(R1), link(R0,D,R1), d(D), closed(D,t-1), not locked(D,t-1). 
% can open a closed but unlocked container
0 {do_open(t,C)} 1 :- at(player,R0,t-1), r(R0), instance_of(C,c), closed(C,t-1), not locked(C,t-1).
% Test constraints
:- do_open(t,CD), d(CD), not closed(CD,t-1). % can't open a door or container that isn't currently closed
:- do_open(t,D), d(D), r(R), atP(t,R), not has_door(R,D).  % can only open a door if player is in appropriate room
% have to be in the same room to open a container
:- do_open(t,C), at(player,R,t), instance_of(C,c), r(R), at(C,R2,t), r(R2), R != R2.

is_action(do_open(t,CD), t) :- do_open(t,CD).  %, is_openable(CD).

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

% - CONSTRAINTS -
:- cookable(X), {raw(X,t); grilled(X,t); fried(X,t); roasted(X,t); burned(X,t) } > 1.   % disjoint set of attribute values for cookable items
:- edible(F,t), inedible(F,t).   % disjoint set of attribute values for potentially edible items

cooked(X,t) :- grilled(X,t).
cooked(X,t) :- fried(X,t).
cooked(X,t) :- roasted(X,t).
cooked(X,t) :- burned(X,t).
inedible(F,t) :- burned(F,t).    % burned foods are considered to be inedible

% --- inertia: cookable items change state only if the player acts on them
needs_cooking(X,t) :- needs_cooking(X,t-1), not act(do_cook(t,X,_),t).
raw(X,t) :- raw(X,t-1), not act(do_cook(t,X,_),t).
grilled(X,t) :- grilled(X,t-1), not act(do_cook(t,X,_),t).
fried(X,t) :- fried(X,t-1), not act(do_cook(t,X,_),t).
roasted(X,t) :- roasted(X,t-1), not act(do_cook(t,X,_),t).
inedible(X,t) :- inedible(X,t-1), not act(do_cook(t,X,_),t).
edible(X,t) :- edible(X,t-1), not act(do_cook(t,X,_),t).
% burned foods stay burned forever after
burned(X,t) :- burned(X,t-1).

burned(X,t) :- cooked(X,t-1), act(do_cook(t,X,_), t).   % cooking something twice causes it to burn
edible(X,t) :- needs_cooking(X,t-1), inedible(X,t-1), not cooked(X,t-1), cooked(X,t). % cooking => transition from inedible to edible
grilled(X,t) :- act(do_cook(t,X,A), t), instance_of(A,toaster), not cooked(X,t-1).  % cooking with a BBQ or grill or toaster
fried(X,t) :- act(do_cook(t,X,A), t), instance_of(A,stove), not cooked(X,t-1).   % cooking on a stove => frying
roasted(X,t) :- act(do_cook(t,X,A), t), instance_of(A,oven), not cooked(X,t-1).   % cooking in an oven => roasting

0 {do_cook(t,X,A)} 1 :- at(player,R,t-1), r(R), cookable(X), instance_of(A,cooker), at(A,R,t-1).
% Test constraints
:- do_cook(t,X,A), atP(R,t), at(A,R2,t), R != R2. % can't cook using an appliance that isn't in the current room



is_action(do_cook(t,X,O), t) :- do_cook(t,X,O).

% ------ CUT ------
% chop :: $in(f, I) & $in(o, I) & $sharp(o) & uncut(f) -> chopped(f)
% dice :: $in(f, I) & $in(o, I) & $sharp(o) & uncut(f) -> diced(f)
% slice :: $in(f, I) & $in(o, I) & $sharp(o) & uncut(f) -> sliced(f)

% ------ CONSTRAINTS ------
:- cuttable(X), {uncut(X,t); chopped(X,t); sliced(X,t); diced(X,t) } > 1.   % disjoint set of attribute values for cuttable items

% --- inertia: cuttable items change state only if the player acts on them
uncut(X,t) :- uncut(X,t-1), not act(do_cut(t,_,X,_),t).

chopped(X,t) :- uncut(X,t-1), act(do_cut(t,chop,X,_),t).
chopped(X,t) :- chopped(X,t-1).
diced(X,t) :- uncut(X,t-1), act(do_cut(t,dice,X,_),t).
diced(X,t) :- diced(X,t-1).
sliced(X,t) :- uncut(X,t-1), act(do_cut(t,slice,X,_),t).
sliced(X,t) :- sliced(X,t-1).

% can chop, slice or dice cuttable ingredients that are in player's inventory if also have a knife (a sharp object), 
0 {do_cut(t,V,F,O):cutting_verb(V) } 1 :- cuttable(F), uncut(F,t-1), in(F,inventory,t-1), sharp(O), in(O,inventory,t-1).

:- do_cut(t,_,F,O), not uncut(F,t-1).  % can't cut up something that's already cut up
:- do_cut(t,_,F,O), not sharp(O).      % can't cut up something with an unsharp instrument

is_action(do_cut(t,V,F,O), t) :- do_cut(t,V,F,O).

"""

GAME_RULES_NEW = \
"""
% ------ TAKE ------
%- take :: $at(P, r) & at(o, r) -> in(o, I)
%+ take :: $at(P, r) & at(o, r) & free(slot) -> in(o, I) & used(slot)",
%- take/c :: $at(P, r) & $at(c, r) & $open(c) & in(o, c) -> in(o, I)
%+ take/c :: $at(P, r) & $at(c, r) & $open(c) & in(o, c) & free(slot) -> in(o, I) & used(slot)",
%- take/s :: $at(P, r) & $at(s, r) & on(o, s) -> in(o, I)
%+ take/s :: $at(P, r) & $at(s, r) & on(o, s) & free(slot) -> in(o, I) & used(slot)",

% ---- inertia: objects don't move unless moved by the player
at(X,R,t) :- at(X,R,t-1), r(R), instance_of(X,thing), not act(do_take(t,X,_),t).
on(X,S,t) :- on(X,S,t-1), s(S), not act(do_take(t,X,_),t).
in(X,C,t) :- in(X,C,t-1), c(C), not act(do_take(t,X,_),t).
in(X,inventory,t) :- in(X,inventory,t-1), not act(do_put(t,X,_),t).

% -- take/c :: can take an object that's in a container if the container is open and player is in the same room
0 {do_take(t,O,C)} 1 :- at(player,R,t-1), r(R), instance_of(O,o), instance_of(C,c), at(C,R,t-1), in(O,C,t-1), open(C,t-1), timestep(t).
% -- take/s :: can take an object that's on a support if player is in the same room as the suppport
0 {do_take(t,O,S)} 1 :- at(player,R,t-1), r(R), instance_of(O,o), s(S), at(S,R,t-1), on(O,S,t-1), timestep(t).
% -- take :: can take an object (a portable thing) if player is in the same room and it is on the floor
0 {do_take(t,O,R)} 1 :- at(player,R,t-1), r(R), instance_of(O,o), at(O,R,t-1), timestep(t).

is_action(do_take(t,O,X),t) :- do_take(t,O,X).
in(O,inventory,t) :- act(do_take(t,O,X),t).  % if player takes an object, it moves to the inventory


% ------ DROP/PUT ------
% put :: $at(P, r) & $at(s, r) & in(o, I) -> on(o, s)
%+ put :: $at(P, r) & $at(s, r) & in(o, I) & used(slot) -> on(o, s) & free(slot)",
% drop :: $at(P, r) & in(o, I) -> at(o, r)
%+ drop :: $at(P, r) & in(o, I) & used(slot) -> at(o, r) & free(slot)",
% insert :: $at(P, r) & $at(c, r) & $open(c) & in(o, I) -> in(o, c)
%+ insert :: $at(P, r) & $at(c, r) & $open(c) & in(o, I) & used(slot) -> in(o, c) & free(slot)",

% insert :: can put an object into a container if the player has the object, container is open and player is in the same room as container
0 {do_put(t,O,C)} 1 :- at(player,R,t-1), r(R), instance_of(O,o), instance_of(C,c), at(C,R,t-1), in(O,inventory,t-1), open(C,t-1), timestep(t).
% put :: can put an object onto a support if player has the object and is in the same room as the suppport
0 {do_put(t,O,S)} 1 :- at(player,R,t-1), r(R), instance_of(O,o), s(S), at(S,R,t-1), in(O,inventory,t-1), timestep(t).
% drop :: can drop an object on the floor of a room if player is in the room and has the object
0 {do_put(t,O,R)} 1 :- at(player,R,t-1), r(R), instance_of(O,o), in(O,inventory,t-1), timestep(t).

is_action(do_put(t,O,X),t) :- do_put(t,O,X).
on(O,X,t) :- act(do_put(t,O,X),t), instance_of(X,s).  % player puts an object onto a supporting object
in(O,X,t) :- act(do_put(t,O,X),t), instance_of(X,c).  % player puts an object into a container
at(O,X,t) :- act(do_put(t,O,X),t), instance_of(X,r).  % player drops an object to the floor of a room


% ------ MAKE ------
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

% --------------------------------------------------------------------------------

"""

# GAME_RULES_OLD = \
# """
# % ------ MAKE ------
# % make/recipe/1 :: $at(P, r) & $cooking_location(r, RECIPE) & in(f, I) & $ingredient_1(f) & $out(meal, RECIPE) -> in(meal, I) & edible(meal) & used(f) & raw(meal)
# % make/recipe/2 :: $at(P, r) & $cooking_location(r, RECIPE) & in(f, I) & $ingredient_1(f) & in(f', I) & $ingredient_2(f') & $out(meal, RECIPE) -> in(meal, I) & edible(meal) & used(f) & used(f') & raw(meal)
# % make/recipe/3 :: $at(P, r) & $cooking_location(r, RECIPE) & in(f, I) & $ingredient_1(f) & in(f', I) & $ingredient_2(f') & in(f'', I) & $ingredient_3(f'') & $out(meal, RECIPE) -> in(meal, I) & edible(meal) & used(f) & used(f') & used(f'') & raw(meal)
# % make/recipe/4 :: $at(P, r) & $cooking_location(r, RECIPE) & in(f, I) & $ingredient_1(f) & in(f', I) & $ingredient_2(f') & in(f'', I) & $ingredient_3(f'') & in(f''', I) & $ingredient_4(f''') & $out(meal, RECIPE) -> in(meal, I) & edible(meal) & used(f) & used(f') & used(f'') & used(f''') & raw(meal)
# % make/recipe/5 :: $at(P, r) & $cooking_location(r, RECIPE) & in(f, I) & $ingredient_1(f) & in(f', I) & $ingredient_2(f') & in(f'', I) & $ingredient_3(f'') & in(f''', I) & $ingredient_4(f''') & in(f'''', I) & $ingredient_5(f'''') & $out(meal, RECIPE) -> in(meal, I) & edible(meal) & used(f) & used(f') & used(f'') & used(f''') & used(f'''') & raw(meal)
# % ------ CONSUME ------
# % drink :: in(f, I) & drinkable(f) -> consumed(f)
# % eat :: in(f, I) & edible(f) -> consumed(f)
# % ------ TAKE ------
# % take :: $at(P, r) & at(o, r) -> in(o, I)
# % take/c :: $at(P, r) & $at(c, r) & $open(c) & in(o, c) -> in(o, I)
# % take/s :: $at(P, r) & $at(s, r) & on(o, s) -> in(o, I)
# % ------ DROP/PUT ------
# % put :: $at(P, r) & $at(s, r) & in(o, I) -> on(o, s)
# % drop :: $at(P, r) & in(o, I) -> at(o, r)
# % insert :: $at(P, r) & $at(c, r) & $open(c) & in(o, I) -> in(o, c)

# """


CHECK_GOAL_ACHIEVED = \
"""
#program check(t).
% Test
solved1(t) :- timestep(T), act(do_examine(T,o_0),T), T < t.
solved2(t) :- solved1(t), act(do_take(t,goal2,_),t), query(t).
%solved2(t) :- solved1(t), diced(goal2,t), query(t).
:- not solved2(t), query(t). % Fail if we haven't achieved all our objectives


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
                aspfile.write(ACTION_STEP_RULES)
                # ---- GAME DYNAMICS
                aspfile.write(GAME_RULES_COMMON)
                aspfile.write(GAME_RULES_NEW)

                aspfile.write(CHECK_GOAL_ACHIEVED)
                #aspfile.write(":- movedP(T,R,R1), at(player,R1,T0), timestep(T0), T0<T .  % disallow loops\n")
                # For branch & bound optimization:
                # aspfile.write( #":- not at(player,r_0,maxT).  % end up in the kitchen\n")
                #     "ngoal(T) :- at(player,R,T), r(R), R!=r_0 .  % want to end up in the kitchen (r_0)\n" \
                #     ":- ngoal(maxT).\n  % anti-goal -- fail if goal not achieved"
                # )
                #aspfile.write("_minimize(1,T) :- ngoal(T).\n")

                #aspfile.write("#show timestep/1.\n")
                #aspfile.write("#show atP/2.\n")
                aspfile.write("#show act/2.\n")
                # aspfile.write("#show at_goal/2.\n")
                # aspfile.write("#show do_moveP/4.\n")
                # aspfile.write("#show do_open/2.\n")
                # aspfile.write("#show has_door/2.\n")
                # aspfile.write("#show openD/2.  % debug \n")

        # Path(destdir).mkdir(parents=True, exist_ok=True)
        # Path(make_dsfilepath(destdir, args.which)).unlink(missing_ok=True)


