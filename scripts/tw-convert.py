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


def twname_to_asp(name:str) -> str:
    if name == 'P':
        aspname = 'player'
    elif name == 'I':
        aspname = 'inventory'
    elif name == 't':  # a base type name in the TextWorld type hierarchy
        aspname = 'thing' 
    else:
        aspname = name.lower()
    return aspname

# def twtype_to_typename(type:str) -> str:
#     return 'thing' if type == 't' else objname_to_asp(type)



#FLUENT_FACTS = [
#   ---location----
#  'at', 'on', 'in',
#   ----open_state----
#  'open', 'closed', 'locked',
#   ----cut_state----
#  'uncut', 'sliced', 'diced', 'chopped',
#   ----cooked_state----
#  'raw', 'needs_cooking', 'roasted', 'grilled', 'fried', 'burned',
#   ----edible_state----
#  'edible', 'inedible']

STATIC_FACTS = ['cuttable', 'cookable', 'sharp', 'cooking_location', 'link', 'north_of', 'east_of', 'south_of', 'west_of']

COOKED_STATE = ['needs_cooking', 'raw', 'roasted', 'grilled', 'fried', 'burned']  #anything other than raw or needs_cooking -> cooked
CUT_STATE = ['uncut', 'sliced', 'diced', 'chopped'] #anything other than -> uncut
OPEN_STATE = ['open', 'closed', 'locked'] #anything other than open -> -open
LOCATION_REL = ['at', 'on', 'in']

def is_recipe_fact(fact_name: str, args_str: Optional[str] = None):
    if fact_name.startswith('ingredient'):
        return True
    if args_str and 'ingredient' in args_str:
        return True
    return False

def is_fluent_fact(fact_name: str, args_str: Optional[str] = None):
    if is_recipe_fact(fact_name, args_str):
        return False
    if fact_name in STATIC_FACTS:
        return False
    return True

def is_state_value(fact_name: str) -> str:
    if fact_name in COOKED_STATE:
        return "cooked_state", "should_cook"
    elif fact_name in CUT_STATE:
        return "cut_state", "should_cut"
    return '', ''

def fact_to_asp(fact:Proposition, hfact=None, step:int = 0) -> str:
    """ converts a TextWorld fact to a format appropriate for Answer Set Programming"""
    asp_fact_str = twfact_to_asp_attrib_state(fact, step)
    if not asp_fact_str:
        asp_args = [Variable(twname_to_asp(v.name), v.type) for v in fact.arguments]
        args_str = ', '.join([f"{a.name}" for a in asp_args])
        maybe_timestep = ', '+str(step) if is_fluent_fact(fact.name, args_str) else ''
        asp_fact_str = f"{fact.name}({args_str}{maybe_timestep})."
    # asp_fact = Proposition(fact.name, asp_args)
    # asp_fact_str = re.sub(r":[^,)]*", '', f"{str(asp_fact)}.")  #remove type annotations (name:type,)
    # if 'ingredient' in asp_fact_str:  #TextWorld quirk: facts refering to 'ingredient_N' are static (define the recipe)
    #     pass   # TODO special-case processing of recipe specifications  
    #     # (don't add time step)
    # elif fact.name not in STATIC_FACTS:
    #     asp_fact_str = asp_fact_str.replace(").", f", {step}).")  # add time step to convert initial state facts to fluents
    if hfact:
        asp_fact_str = f"{asp_fact_str} % {str(hfact)}"  # human readable version of the fact (obj names instead of ids)
    return asp_fact_str

def twfact_to_asp_attrib_state(fact:Proposition, step:int):
    attrib_name, attrib_name_should = is_state_value(fact.name)
    if not attrib_name:
        return None
    assert fact.name not in STATIC_FACTS, f"UNEXPECTED: {fact.name} is BOTH in STATIC_FACTS and an attribute_value[{attrib_name}]"
    asp_args = [Variable(twname_to_asp(v.name), v.type) for v in fact.arguments]
    arg_names = [f"{a.name}" for a in asp_args]
    assert len(arg_names) == 1, f"{arg_names}"
    is_recipe_entry =  is_recipe_fact(fact.name, args_str=arg_names[0])  # if multiple args, need to do a string join
    arg_names.append(fact.name) 
    if is_recipe_entry: 
        #maybe_timestep = ''   # recipe is static, no timestep needed
        attrib_name = attrib_name_should
    else:
        arg_names.append(str(step))   # add timestep as last arg to make a fluent
    args_str = ', '.join(arg_names)
    asp_fact_str = f"{attrib_name}({args_str})."
    return asp_fact_str

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
"""% #include <incmode>.
#const imax=500.  % (default value for) give up after this many iterations
#script (python)

from clingo import Function, Symbol, String, Number

def get(val, default):
    return val if val != None else default

def main(prg):
    imin = get(prg.get_const("imin"), Number(1))
    imax = get(prg.get_const("imax"), Number(500))
    istop = get(prg.get_const("istop"), String("SAT"))

    step, ret = 0, None
    while ((imax is None or step < imax.number) and
           (step == 0 or step <= imin.number or (
              (istop.string == "SAT" and not ret.satisfiable) or
              (istop.string == "UNSAT" and not ret.unsatisfiable) or
              (istop.string == "UNKNOWN" and not ret.unknown))
           )):
        parts = []
        parts.append(("check", [Number(step)]))
        if step > 0:
            prg.release_external(Function("query", [Number(step-1)]))
            parts.append(("step", [Number(step)]))
            #? prg.cleanup()
        else:
            parts.append(("base", []))
        prg.ground(parts)
        prg.assign_external(Function("query", [Number(step)]), True)
        ret, step = prg.solve(), step+1
#end.

#program check(t).
#external query(t).

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

% additional inheritance links to simplify rules for attributes related to cooked_state and cut_state
class(attribute_value).
subclass_of(cooked_state, attribute_value).
subclass_of(cut_state, attribute_value).

%{is_openable(X); is_lockable(X)}=2 :- instance_of(X,d). % doors are potentially openable and lockable
is_openable(X) :- instance_of(X,d).  % doors are potentially openable
is_lockable(X) :- instance_of(X,d).  % doors are potentially lockable
is_openable(X) :- instance_of(X,c).  % containers are potentially openable
is_lockable(X) :- instance_of(X,c), not instance_of(X,oven). % most containers are potentially lockable, but ovens are not

% action vocabulary
timestep(0). % incremental solving will define timestep(t) for t >= 1...

direction(east;west;north;south).

cutting_verb(chop;slice;dice).

cooked_state(needs_cooking;raw;grilled;roasted;fried;burned).
cut_state(uncut;chopped;diced;sliced).

instance_of(X,cooked_state) :- cooked_state(X).
instance_of(X,cut_state) :- cut_state(X).

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
:- cookable(X), {cooked_state(X,V,t):instance_of(V,cooked_state)} > 1.   % disjoint set of attribute values for cookable items
:- edible(F,t), inedible(F,t).   % disjoint set of attribute values for potentially edible items

cooked(X,t) :- cooked_state(X,V,t), V != raw, V != needs_cooking, instance_of(V,cooked_state).
inedible(F,t) :- cooked_state(F,burned,t).    % burned foods are considered to be inedible

% --- inertia: cookable items change state only if the player acts on them
cooked_state(X,V,t) :- cooked_state(X,V,t-1), not act(do_cook(t,X,_),t).
% burned foods stay burned forever after
cooked_state(X,burned,t) :- cooked_state(X,burned,t-1).
:- cooked_state(X,S1,t), S1 != burned, cooked_state(X,burned,t-1).

cooked_state(X,grilled,t) :- act(do_cook(t,X,A), t), instance_of(A,toaster), not cooked(X,t-1).  % cooking with a BBQ or grill or toaster
cooked_state(X,fried,t) :- act(do_cook(t,X,A), t), instance_of(A,stove), not cooked(X,t-1).   % cooking on a stove => frying
cooked_state(X,roasted,t) :- act(do_cook(t,X,A), t), instance_of(A,oven), not cooked(X,t-1).   % cooking in an oven => roasting
cooked_state(X,burned,t) :- cooked(X,t-1), act(do_cook(t,X,_), t).   % cooking something twice causes it to burn

inedible(X,t) :- inedible(X,t-1), not act(do_cook(t,X,_),t).
edible(X,t) :- edible(X,t-1), not act(do_cook(t,X,_),t).
%edible(X,t) :- cooked_state(X,needs_cooking,t-1), inedible(X,t-1), not cooked(X,t-1), cooked(X,t). % cooking => transition from inedible to edible
edible(X,t) :- cooked(X,t), cooked_state(X,V,t), V!=burned. % cooking => transition from inedible to edible

0 {do_cook(t,X,A)} 1 :- at(player,R,t-1), r(R), cookable(X), instance_of(A,cooker), in_inventory(X,t-1), at(A,R,t-1).
% Test constraints
:- do_cook(t,X,A), atP(R,t), at(A,R2,t), R != R2. % can't cook using an appliance that isn't in the current room

is_action(do_cook(t,X,O), t) :- do_cook(t,X,O).

% ------ CUT ------
% chop :: $in(f, I) & $in(o, I) & $sharp(o) & uncut(f) -> chopped(f)
% dice :: $in(f, I) & $in(o, I) & $sharp(o) & uncut(f) -> diced(f)
% slice :: $in(f, I) & $in(o, I) & $sharp(o) & uncut(f) -> sliced(f)

% ------ CONSTRAINTS ------
:- cuttable(X), {cut_state(X,V,t):instance_of(V,cut_state) } > 1.   % disjoint set of attribute values for cuttable items

% --- inertia: cuttable items change state only if the player acts on them
cut_state(X,uncut,t) :- cut_state(X,uncut,t-1), not act(do_cut(t,_,X,_),t).
cut_state(X,chopped,t) :- cut_state(X,uncut,t-1), act(do_cut(t,chop,X,_),t).
cut_state(X,diced,t) :- cut_state(X,uncut,t-1), act(do_cut(t,dice,X,_),t).
cut_state(X,sliced,t) :- cut_state(X,uncut,t-1), act(do_cut(t,slice,X,_),t).

% cut-up items remain cut-up, and can't be cut up any further
cut_state(X,V,t) :- cut_state(X,V,t-1), V != uncut.

% can chop, slice or dice cuttable ingredients that are in player's inventory if also have a knife (a sharp object), 
0 {do_cut(t,V,F,O):cutting_verb(V) } 1 :- cuttable(F), cut_state(F,uncut,t-1), in(F,inventory,t-1), sharp(O), in(O,inventory,t-1), not cooked(F,t-1).

:- do_cut(t,_,F,O), not cut_state(F,uncut,t-1).  % can't cut up something that's already cut up
:- do_cut(t,_,F,O), not sharp(O).      % can't cut up something with an unsharp instrument
:- do_cut(t,_,F,_), cooked(F,t).       % can't cut up an ingredient that has already been cooked (in TextWorld)

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
on(X,S,t) :- on(X,S,t-1), instance_of(S,s), not act(do_take(t,X,_),t).
in(X,C,t) :- in(X,C,t-1), instance_of(C,c), not act(do_take(t,X,_),t).
in(X,inventory,t) :- in(X,inventory,t-1), not act(do_put(t,X,_),t), not consumed(X,t).

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

in_recipe(I,F) :- ingredient(I), in(I,recipe), base(F,I), instance_of(F,f).
in_recipe(F) :- in_recipe(I,F).

% have_prepped_ingredients is True if all required ingredients have been fully prepared and are currently in player's inventory
0 { have_prepped_ingredients(t) } 1 :- in_recipe(I,F), in_inventory(F,t), timestep(t).
:- have_prepped_ingredients(t), in_recipe(F), not in_inventory(F,t), timestep(t).
:- have_prepped_ingredients(t), in_recipe(I,F), should_cook(I,V), cookable(F), not cooked_state(F,V,t), timestep(t).
:- have_prepped_ingredients(t), in_recipe(I,F), should_cut(I,V), cuttable(F), not cut_state(F,V,t), timestep(t).

0 { do_make_meal(t) } 1 :- have_prepped_ingredients(t-1), cooking_location(R, recipe), r(R), atP(t,R), timestep(t).
:- do_make_meal(t), cooking_location(R, recipe), r(R), not atP(t,R), timestep(t).

is_action(do_make_meal(t),t) :- do_make_meal(t), timestep(t).

in(meal_0,inventory,t) :- act(do_make_meal(t),t), timestep(t).
consumed(F,t) :- act(do_make_meal(t),t), in_recipe(F), timestep(t).

% an example of how to count something - number of recipe ingredients that are currently in our inventory
% NOTE: INEFFICIENT - NOTICEABLY SLOWS DOWN INCREMENTAL SOLVING (if done at each step)
%num_acquired(t,N) :- timestep(t), N=#count{F:in_recipe(I,F),in_inventory(F,t),instance_of(F,f),instance_of(I,ingredient)}. %, query(t).
%#show num_acquired/2.


% ------ CONSUME ------
% drink :: in(f, I) & drinkable(f) -> consumed(f)
%+ drink :: in(f, I) & drinkable(f) & used(slot) -> consumed(f) & free(slot)",
% eat :: in(f, I) & edible(f) -> consumed(f)
%+ eat :: in(f, I) & edible(f) & used(slot) -> consumed(f) & free(slot)",

0 {do_eat(t,F)} 1 :- edible(F,t-1), instance_of(F,f), in_inventory(F,t-1), timestep(t).
0 {do_drink(t,F)} 1 :- drinkable(F,t-1), instance_of(F,f), in_inventory(F,t-1), timestep(t).

is_action(do_eat(t,F),t) :- do_eat(t,F), instance_of(F,f), timestep(t).
is_action(do_drink(t,F),t) :- do_drink(t,F), instance_of(F,f), timestep(t).

consumed(F,t) :- act(do_eat(t,F),t).
consumed(F,t) :- act(do_drink(t,F),t).

consumed(F,t) :- consumed(F,t-1), timestep(t).

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

recipe_seen(t) :- act(do_examine(t,o_0),t), timestep(t).
recipe_seen(t) :- recipe_seen(t-1), timestep(t).

%--------------------
:- not recipe_seen(t), query(t).
%solved2(t) :- act(do_make_meal(t),t), query(t).
solved2(t) :- consumed(meal_0,t), query(t).
%--------------------
% solved1(t) :- recipe_seen(t), query(t).
% solved2(t) :- solved1(t), have_prepped_ingredients(t), query(t).
%--------------------

:- not solved2(t), query(t).


"""


def types_to_asp(typestree: VariableTypeTree) -> str:
    # typestree.serialize(): return [vtype.serialize() for vtype in self.variables_types.values()]
    def _vtype_info(vtype:Variable) -> str:
        info_tuple = (twname_to_asp(vtype.name), twname_to_asp(vtype.parent) if vtype.parent else None)
        return info_tuple
    type_infos = [_vtype_info(vtype) for vtype in typestree.variables_types.values()]
    return type_infos


def info_to_asp(info) -> str:
    type_name = twname_to_asp(info.type)
    info_type_str = f"{type_name}({twname_to_asp(info.id)})."
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


