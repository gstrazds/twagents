import re
#from pathlib import Path, PurePath
from typing import Optional, Any, Tuple
from operator import itemgetter

from textworld.logic import Proposition, Variable
from textworld.generator import Game, KnowledgeBase
from textworld.generator.vtypes import VariableType, VariableTypeTree
from textworld.generator.inform7 import Inform7Game


INCREMENTAL_SOLVING = \
"""% #include <incmode>.
#const imax=500.  % (default value for) give up after this many iterations
#const find_first=o_0.  % the cookbook is always o_0
%#const first_room=r_4.  %this is a temporary hack [can override from command line with -c first_room=r_N]
#script (python)
import datetime
from clingo import Function, Symbol, String, Number, SymbolType

def get(val, default):
    return val if val != None else default

def main(prg):

    imin = get(prg.get_const("imin"), Number(1))
    imax = get(prg.get_const("imax"), Number(500))
    istop = get(prg.get_const("istop"), String("SAT"))
    #first_room = get(prg.get_const("first_room"), String("r_4"))

    _actions_list = []
    _actions_facts = []
    _newly_discovered_facts = []  # rooms or opened containers
    _recipe_read = False
    def _get_chosen_actions(model, step):
        #for act in prg.symbolic_atoms.by_signature("act",2):
        #     print(f"[t={act.symbol.arguments[1].number}] action:{act.symbol.arguments[0]}")
        print(f"_get_chosen_actions(model,{step})......")
        _actions_list.clear()
        _actions_facts.clear()
        _newly_discovered_facts.clear()
        _solved_all = False
        for atom in model.symbols(atoms=True):
            if (atom.name == "act" and len(atom.arguments)==2 and atom.arguments[1].type is SymbolType.Number):
                t = atom.arguments[1].number
                action = atom.arguments[0]
                str_atom = f"{atom}."
                _actions_facts.append(str_atom)
                print(f"  -- [{t}] : {str_atom}")
                _actions_list.append(atom)
            elif atom.name == 'recipe_read' \\
              and len(atom.arguments)==1 and atom.arguments[0].type is SymbolType.Number:
                if atom.arguments[0].number == step:
                    print(f"  -- {atom}")
                _newly_discovered_facts.append("cookbook")
            elif atom.name == 'solved_all' \\
              and len(atom.arguments)==1 and atom.arguments[0].type is SymbolType.Number:
                print(f"  -- {atom}")
                _newly_discovered_facts.append("solved_all")
                _solved_all = True
            elif (atom.name == 'first_visited' \\
               or atom.name == 'first_opened'  \\
               or atom.name == 'first_acquired') \\
              and len(atom.arguments)==2 and atom.arguments[1].type is SymbolType.Number:
                if atom.arguments[1].number == step:    #, f"[{step}] {atom.name} {atom.arguments[1]}"
                    print(f"  -- {atom}")
                    _newly_discovered_facts.append(str(atom.arguments[0]))
        if _solved_all:
            return False
        return True  # False -> stop after first model

    MIN_PRINT_STEP = 0  # print elapsed time for each solving step >= this value
    MAX_STEP_ELAPSED_TIME = datetime.timedelta(minutes=1, seconds=20)
    step, ret = 0, None
    solved_all = False
    while ((imax is None or step < imax.number) and
           (step == 0 or step <= imin.number or (
              (istop.string == "SAT" and (not ret.satisfiable or not solved_all)
               or (istop.string == "UNSAT" and not ret.unsatisfiable)
               or (istop.string == "UNKNOWN" and not ret.unknown)
              )
           ))):
        start_time = datetime.datetime.now()
        _recipe_newly_seen = False  # becomes True during the step that finds the recipe
        parts = []
        if step >= MIN_PRINT_STEP:
            print(f"solving:[{step}] ", end='', flush=True)
        if step == 0:
            parts.append(("base", []))
            parts.append(("recipe", [Number(step)]))
            parts.append(("initial_state", [Number(0)]))
            parts.append(("initial_room", [Number(0)]))     #(f"room_{first_room}", [Number(0)]))
            parts.append(("every_step", [Number(step)]))
            parts.append(("check", [Number(step)]))
        else:  #if step > 0:
            #  query(t-1) becomes permanently = False (removed from set of Externals)
            if ret and ret.satisfiable:
                for _name in _newly_discovered_facts:
                    if _name == "solved_all":
                        solved_all = True
                    elif _name == "cookbook":
                        if not _recipe_read:
                            _recipe_newly_seen = True
                            print(f"+++++ _recipe_newly_seen: (NOT ADDING) #program recipe({step-1})")
                            #parts.append(("recipe", [Number(step-1)]))
                            parts.append(("cooking_step", [Number(step-1)]))  # this once will have cooking_step(both step and step-1)
                        _recipe_read = True
                    elif _name.startswith("r_"):    # have entered a previously unseen room
                        print(f"ADDING #program room_{_name} ({step-1}).")
                        parts.append((f"room_{_name}", [Number(step-1)]))
                    else:
                        print("%%%%% IGNORING FIRST OPENED or ACQUIRED:", _name)
                if solved_all:
                    break      # stop solving immediately

            if len(_actions_facts):
                actions_facts_str = "\\n".join(_actions_facts)
                actions_facts_str = actions_facts_str.replace("act(", "did_act(")
                print(f"\\n+++++ ADDING prev_actions: +++\\n{actions_facts_str}\\n----", flush=True)
                print("\\t", "\\n\\t".join(_actions_facts), flush=True)
                prg.add("prev_actions", [], actions_facts_str)
                parts.append(("prev_actions", []))

            parts.append(("every_step", [Number(step)]))
            parts.append(("step", [Number(step)]))
            if _recipe_read:
                print(f"+ ADDING #program cooking_step({step})")
                parts.append(("cooking_step", [Number(step)]))
            parts.append(("check", [Number(step)]))

            prg.release_external(Function("query", [Number(step-1)]))
            prg.cleanup()
        prg.ground(parts)

        prg.assign_external(Function("query", [Number(step)]), True)
 
        ret = prg.solve(on_model=lambda model: _get_chosen_actions(model,step))
        finish_time = datetime.datetime.now()
        elapsed_time = finish_time-start_time
        print("<< SATISFIABLE >>" if ret.satisfiable else "<< NOT satisfiable >>", flush=True)
        if step >= MIN_PRINT_STEP:
            print(f"--- [{step}] elapsed: {elapsed_time}")
        if elapsed_time > MAX_STEP_ELAPSED_TIME:
            print(f"--- [{step}] Exceded step time threshhold({MAX_STEP_ELAPSED_TIME})... Stop solving.")
            break
        step = step+1

#end.

#program check(t).
#external query(t).
%#program check2(t).
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
unknown(unknown).

%inventory_max(7) :- not class(slot).  % default value for inventory size for games with no explicit inventory logic (old FTWC games)
inventory_max(N) :- class(slot), N=#count {slot(X):slot(X)}.

cutting_verb(chop;slice;dice).

cooked_state(needs_cooking;raw;grilled;roasted;fried;burned).
cut_state(uncut;chopped;diced;sliced).

instance_of(X,cooked_state) :- cooked_state(X).
instance_of(X,cut_state) :- cut_state(X).

% additional inheritance links to simplify rules for unknown rooms
unknown(unknownN;unknownS;unknownE;unknownW).  % distinct unknowns for each cardinal direction (N,S,E,W)
instance_of(unknownN;unknownS;unknownE;unknownW, unknown).


%---- initialize step 0 values for some fluents that require inertia from step-1

need_to_find(find_first,0).  % initially, we're looking for the cookbook

%have_found(R,0) :- r(R), at(player,R,0). % have found the initial room (NOT ANYMORE: first room added dynamically at step 1)

% can see a thing if player is in the same room as the thing
%have_found(O,0) :- at(player,R,0), r(R), instance_of(O,thing), at(O,R,0).
% can see an object that's in a container if the container is open and player is in the same room
%have_found(O,0) :- at(player,R,0), r(R), instance_of(O,o), instance_of(C,c), at(C,R,0), in(O,C,0), open(C,0).
% can see an object that is on a support if player is in the same room as the suppport
%have_found(O,0) :- at(player,R,0), r(R), instance_of(O,o), instance_of(S,s), at(S,R,0), on(O,S,0).
% have already found something if it is initially in the player's inventory
%have_found(O,0) :- in(O,inventory,0).
%first_visited(R,0) :- r(R), at(player,R,0).

contents_unknown(C,0) :- instance_of(C,c), closed(C,0).  % can't see into closed containers

% privileged (fully observable ground truth) knownledge about room connectivity
% is needed to emulate the navigation transitions of the game (when running standalone - not embedded with game engine)
_connected(R1,R2,east) :- _west_of(R1, R2), r(R1), r(R2).
_connected(R1,R2,south) :- _north_of(R1, R2), r(R1), r(R2).
_connected(R1,R2,north) :- _south_of(R1, R2), r(R1), r(R2).
_connected(R1,R2,west) :- _east_of(R1, R2), r(R1), r(R2).
% assume that all doors/exits can be traversed in both directions
_connected(R1,R2,east) :- _connected(R2,R1,west), r(R1).
_connected(R1,R2,west) :- _connected(R2,R1,east), r(R1).
_connected(R1,R2,south) :- _connected(R2,R1,north), r(R1).
_connected(R1,R2,north) :- _connected(R2,R1,south), r(R1).

"""
#  "class(A) :- instance_of(I,A).

EVERY_STEP = \
"""
% ------------------ every_step(t) t=[0...] ----------------------
#program every_step(t).  % fluents that are independent of history (don't reference step-1, apply also for step 0)

% player has visited room R
have_found(R,t) :- r(R), at(player,R,t).
first_found(X,t) :- have_found(X,t), not have_found(X,t-1).
first_visited(R,t) :- r(R), first_found(R,t).

have_found(O,t) :- at(player,R,t), r(R), instance_of(O,o), in(O,inventory,t), timestep(t).
% can see a thing if player is in the same room as the thing
have_found(O,t) :- instance_of(O,thing), is_here(O,t), timestep(t).
% can see an object that is on a support if player is in the same room as the suppport
%have_found(O,t) :- at(player,R,t), r(R), at(S,R,t), on(O,S,t), timestep(t).
% can see an object that's in a container if the container is open and player is in the same room
%have_found(O,t) :- at(player,R,t), r(R), instance_of(O,o), instance_of(C,c), at(C,R,t), in(O,C,t), open(C,t), timestep(t).

need_to_search(t) :- {not have_found(X,t):need_to_find(X,t)}>0.

need_to_gather(t) :- {need_to_acquire(X,t):need_to_acquire(X,t)}>0.

atP(R,t) :- at(player,R,t), r(R).   % Alias for player's initial position"
in_inventory(O,t) :- in(O,inventory,t).      % alias for object is in inventory at time t
"""

# NOTE: the following MAP_RULES are also evaluated at every step
EVERY_STEP_MAP_RULES = \
"""

% ------- Navigation (evaluated at every step: eventually grounded for all room combinations R1xR2) -------
connected(R1,unknownE,east) :- east_of(unknownE, R1), r(R1).
connected(R1,unknownS,south) :- south_of(unknownS, R1), r(R1).
connected(R1,unknownN,north) :- north_of(unknownN, R1), r(R1).
connected(R1,unknownW,west) :- west_of(unknownW, R1), r(R1).


link(R1,D,R2) :- r(R1), r(R2), d(D), link(R1,D,unknownE), link(R2,D,unknownW).
link(R1,D,R2) :- r(R1), r(R2), d(D), link(R1,D,unknownW), link(R2,D,unknownE).
link(R1,D,R2) :- r(R1), r(R2), d(D), link(R1,D,unknownN), link(R2,D,unknownS).
link(R1,D,R2) :- r(R1), r(R2), d(D), link(R1,D,unknownS), link(R2,D,unknownN).

%connected(R1,R2) :- connected(R1,R2,NSEW), direction(NSEW).
%connected(R1,R2) :- connected(R1,R2,NSEW,t), r(R2), direction(NSEW).
%has_door(R,D) :- r(R), d(D), link(R,D,_).
has_door(R,D) :- r(R), d(D), direction(NSEW), link(R,D,R2), connected(R,R2,NSEW).
door_direction(R,D,NSEW) :- r(R), d(D), direction(NSEW), link(R,D,R2), connected(R,R2,NSEW).

% define fluents that determine whether player can move from room R0 to R1
%%%%%%%% TODO: ? add NSEW to free(R0,R1,NSEW,t)

free(R0,R1,t) :- r(R0), d(D), link(R0,D,R1), open(D,t). %, r(R1)
not free(R0,R1,t) :- r(R0), d(D), link(R0,D,R1), not open(D,t). %, r(R1)

%free(R0,R1,t) :- r(R0), connected(R0,R1), not link(R0,_,R1).  %, r(R1), % if there is no door, exit is always traversible.
free(R0,R1,t) :- r(R0), connected(R0,R1,NSEW), direction(NSEW), not link(R0,_,R1).  %, r(R1), % if there is no door, exit is always traversible.
free(R0,R1,t) :- r(R0), connected(R0,R1,NSEW,t), not door_direction(R0,_,NSEW).  %, r(R1), % if there is no door, exit is always traversible.

connected(R1,R2,east,t) :- act(do_moveP(t,R1,unknownE,east),t), at(player,R2,t), r(R2), R1!=R2.
connected(R1,R2,south,t) :- act(do_moveP(t,R1,unknownS,south),t), at(player,R2,t), r(R2), R1!=R2.
connected(R1,R2,north,t) :- act(do_moveP(t,R1,unknownN,north),t), at(player,R2,t), r(R2), R1!=R2.
connected(R1,R2,west,t) :- act(do_moveP(t,R1,unknownW,west),t), at(player,R2,t), r(R2), R1!=R2.

% assume that all doors/exits can be traversed in both directions
% assume that all doors/exits can be traversed in both directions
connected(R1,R2,east,t) :- connected(R2,R1,west,t), r(R1).
connected(R1,R2,west,t) :- connected(R2,R1,east,t), r(R1).
connected(R1,R2,south,t) :- connected(R2,R1,north,t), r(R1).
connected(R1,R2,north,t) :- connected(R2,R1,south,t), r(R1).

% is the thing X close enough to the player to interact with
in_room(X,R,t) :- r(R), r(X), X=R.
in_room(X,R,t) :- at(X,R,t), r(R).
in_room(X,R,t) :- on(X,S,t), in_room(S,R,t), instance_of(S,s), r(R).
in_room(X,R,t) :- in(X,C,t), in_room(C,R,t), instance_of(C,c), open(C,t), r(R).

is_here(X,t) :- at(player,R,t), in_room(X,R,t), r(R).
can_take(O,t) :- instance_of(O,o), is_here(O,t). 

"""

ACTION_STEP_RULES = \
"""
% ------------------ step(t) t=[1...] ----------------------
#program step(t).    % applied at each timestep >=1

% Generate
timestep(t).



% inertia
connected(R1,R2,NSEW,t) :- connected(R1,R2,NSEW,t-1),r(R1),r(R2),direction(NSEW). % once connected -> always connected
free(R0,R1,t) :- free(R0,R1,t-1), not link(R0,_,R1).  % no door -> can never become closed/unpassable

{act(X,t):is_action(X,t)} = 1 :- timestep(t). % player must choose exactly one action at each time step.

{at(player,R,t):r(R)} = 1 :- timestep(t).   % player is in exactly one room at any given time
% NOTE/IMPORTANT - THE FOLLOWING IS NOT THE SAME as prev line, DOES NOT WORK CORRECTLY:
%  {at(player,R,t)} = 1 :- r(R), timestep(t).
% NOTE - THE FOLLOWING ALSO DOESN'T WORK (expands too many do_moveP ground instances at final timestep:
%  {at(player,R,t)} = 1 :- r(R), at(player,R,t), timestep(t).

need_to_find(X,t) :- need_to_find(X,t-1), not have_found(X,t-1), timestep(t).
need_to_find(X,t) :- need_to_acquire(X,t), not have_found(X,t).
%:- need_to_find(X,t), have_found(X,t-1).

need_to_acquire(X,t) :- need_to_acquire(X,t-1), instance_of(X,o), not have_acquired(X,t-1), timestep(t).
:- need_to_acquire(X,t), have_acquired(X,t-1).   % if we've gotten it once, deprioritize it
have_acquired(X,t) :- have_acquired(X,t-1).
have_acquired(X,t) :- in(X,inventory,t), need_to_acquire(X,t-1), timestep(t).

first_acquired(X,t) :- have_acquired(X,t), not have_acquired(X,t-1), timestep(t).

contents_unknown(C,t) :- contents_unknown(C,t-1), timestep(t), closed(C,t).  % not act(do_open(t,C),t).

% newly explored a container (that was previously closed)
first_opened(C,t) :- contents_unknown(C,t-1), not contents_unknown(C,t), timestep(t).


% Alias
%MOVED TO every_step(t) -- atP(R,t) :- at(player,R,t).                 % alias for player's current location
%in_inventory(O,t) :- in(O,inventory,t).      % alias for object is in inventory at time t

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

%inertia: once we've found something, it stays found forever
% because the player is the only agent in a deterministic world, with perfect memory
%T1=T1 :- have_found(O,T1,t), have_found(O,T2,t-1).
have_found(X,t) :- have_found(X,t-1).

0 {do_look(t,R)} 1 :- at(player,R,t), r(R), timestep(t).

0 {do_examine(t,O)} 1 :- is_here(O,t), instance_of(O,thing), timestep(t).
0 {do_examine(t,O)} 1 :- at(player,R,t), r(R), instance_of(O,o), in(O,inventory,t), timestep(t).

% examine/c :: can examine an object that's in a container if the container is open and player is in the same room
%0 {do_examine(t,O)} 1 :- at(player,R,t), r(R), instance_of(O,o), instance_of(C,c), at(C,R,t), in(O,C,t), open(C,t), timestep(t).
% examine/s :: can examine an object that is on a support if player is in the same room as the suppport
%0 {do_examine(t,O)} 1 :- at(player,R,t), r(R), at(S,R,t), on(O,S,t), timestep(t).
% examine/t :: can examine a thing if player is in the same room as the thing
%0 {do_examine(t,O)} 1 :- at(player,R,t), r(R), instance_of(O,thing), at(O,R,t), timestep(t).

% Test constraints
% have to be in the same room to examine something
:- do_examine(t,O), at(player,R,t), o(O), r(R), on(O,S,t), at(S,R2,t), s(S), r(R2), timestep(t), R != R2.
:- do_examine(t,O), at(player,R,t), o(O), r(R), in(O,C,t), at(C,R2,t), c(C), r(R2), timestep(t), R != R2.
:- do_examine(t,O), at(player,R,t), o(O), r(R), at(O,R2,t), r(R2), timestep(t), R != R2.

is_action(do_examine(t,O), t) :- do_examine(t,O). %, instance_of(O,thing).
is_action(do_look(t,R), t) :- do_look(t,R). %, r(R).

have_examined(O,t,t) :- act(do_examine(t,O),t), instance_of(O,thing), timestep(t).
have_examined(R,t,t) :- act(do_look(t,R),t), r(R), timestep(t).

% inertia
have_examined(X,T,t) :- have_examined(X,T,t-1), timestep(t), timestep(T), T<t.  %, not act(do_examine(t,X),t), not act(do_look(t,X),t).

recipe_read(t) :- have_examined(o_0, _, t).   % o_0 is always the RECIPE
% recipe_read(t) :- act(do_examine(t,o_0),t), timestep(t).   % o_0 is always the RECIPE
% recipe_read(t) :- recipe_read(t-1), timestep(t).


% ------ GO ------
% go/east :: at(P, r) & $west_of(r, r') & $free(r, r') & $free(r', r) -> at(P, r')
% go/north :: at(P, r) & $north_of(r', r) & $free(r, r') & $free(r', r) -> at(P, r')
% go/south :: at(P, r) & $north_of(r, r') & $free(r, r') & $free(r', r) -> at(P, r')
% go/west :: at(P, r) & $west_of(r', r) & $free(r, r') & $free(r', r) -> at(P, r')

% --- inertia: player stays in the current room unless acts to move to another
  % stay in the current room unless current action is do_moveP
%at(player,R0,t) :- at(player,R0,t-1), r(R0), {act(do_moveP(t,R0,R,NSEW),t):r(R),direction(NSEW)}=0. %, T<=maxT.
at(player,R0,t) :- at(player,R0,t-1), r(R0), not act(do_moveP(t,R0,_,NSEW),t):direction(NSEW). %, T<=maxT.

  % player moved at time t, from previous room R0 to new room R
at(player,R,t) :- act(do_moveP(t,R0,_,NSEW),t), at(player,R0,t-1), r(R0), _connected(R0,R,NSEW), direction(NSEW). %, R!=R0.

% Test constraints
%:- at(player,R0,t-1), at(player,R,t), r(R0), r(R), R!=R0, not free(R,R0,t-1).

 % can move to a connected room, if not blocked by a closed door
0 {do_moveP(t,R0,R,NSEW):free(R0,R,t-1),connected(R0,R,NSEW),direction(NSEW)} 1 :- at(player,R0,t-1), free(R0,R,t-1), direction(NSEW), r(R0). %
0 {do_moveP(t,R0,R,NSEW):free(R0,R,t-1),connected(R0,R,NSEW,t-1),direction(NSEW)} 1 :- at(player,R0,t-1), free(R0,R,t-1), direction(NSEW), r(R0). %


% Test constraints
:- do_moveP(t,R0,R,NSEW),direction(NSEW),r(R0),r(R),timestep(t),not free(R0,R,t-1).  % can't go that way: not a valid action
:- do_moveP(t,R0,U,NSEW),unknown(U),direction(NSEW),r(R0),timestep(t),not free(R0,U,t-1).  % can't go that way: not a valid action
:- do_moveP(t,R0,U,NSEW),unknown(U),direction(NSEW),r(R0),timestep(t),r(R2),at(player,R2,t),free(R0,R2,t-1).  % if dest room is known, use it explicitly


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
open(X,t) :- is_openable(X), open(X,t-1), not act(do_close(t,X),t).
open(X,t) :- is_openable(X), act(do_open(t,X),t).
closed(X,t) :- closed(X,t-1), not act(do_open(t,X),t).
locked(X,t) :- locked(X,t-1), not act(do_unlock(t,X,_),t).
% ------ CONSTRAINTS ------
:- open(X,t), closed(X,t).  %[,is_openable(X).]    % any door or container can be either open or closed but not both
:- locked(X,t), open(X,t).  %[,is_lockable(X).]    % can't be both locked and open at the same time

% can open a closed but unlocked door
0 {do_open(t,D)} 1 :- at(player,R0,t), r(R0), link(R0,D,R1), d(D), closed(D,t-1), not locked(D,t-1). % R1 -- might be unknown: = not a room
% can open a closed but unlocked container
0 {do_open(t,C)} 1 :- at(player,R0,t-1), r(R0), instance_of(C,c), closed(C,t-1), not locked(C,t-1).
% Test constraints
:- do_open(t,CD), d(CD), not closed(CD,t-1). % can't open a door or container that isn't currently closed
:- do_open(t,D), d(D), r(R), at(player,R,t), not has_door(R,D).  % can only open a door if player is in appropriate room
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

0 {do_cook(t,X,A)} 1 :- at(player,R,t-1), r(R), cookable(X), instance_of(A,cooker), in(X,inventory,t-1), at(A,R,t-1).
% Test constraints
:- do_cook(t,X,A), at(player,R,t), at(A,R2,t), R != R2. % can't cook using an appliance that isn't in the current room

is_action(do_cook(t,X,A), t) :- do_cook(t,X,A).

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
0 {do_cut(t,V,F,O):cutting_verb(V) } 1 :- cuttable(F), cut_state(F,uncut,t-1), in(F,inventory,t-1), sharp(O), in(O,inventory,t-1). %, not cooked(F,t-1).
%:- do_cut(t,_,F,_), cooked(F,t).       % can't cut up an ingredient that has already been cooked (in TextWorld)

:- do_cut(t,_,F,O), not cut_state(F,uncut,t-1).  % can't cut up something that's already cut up
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
on(X,S,t) :- on(X,S,t-1), instance_of(S,s), not act(do_take(t,X,_),t).
in(X,C,t) :- in(X,C,t-1), instance_of(C,c), not act(do_take(t,X,_),t).
in(X,inventory,t) :- in(X,inventory,t-1), not act(do_put(t,X,_),t), not consumed(X,t).

% -- take/c :: can take an object that's in a container if the container is open and player is in the same room
0 {do_take(t,O,C)} 1 :- at(player,R,t-1), r(R), instance_of(O,o), instance_of(C,c), at(C,R,t-1), in(O,C,t-1), open(C,t-1), timestep(t).
% -- take/s :: can take an object that's on a support if player is in the same room as the suppport
0 {do_take(t,O,S)} 1 :- at(player,R,t-1), r(R), instance_of(O,o), instance_of(S,s), at(S,R,t-1), on(O,S,t-1), timestep(t).

% -- take :: can take an object (a portable thing) if player is in the same room and it is on the floor
0 {do_take(t,O,R)} 1 :- at(player,R,t-1), r(R), instance_of(O,o), at(O,R,t-1), timestep(t).

%count_inventory(N,t) :- timestep(t), N=#sum{1,in(O,inventory,t):in(O,inventory,t)}.  % #sum of 1, = same as #count{}
%#show count_inventory/2.

% can't pick up anything if already carrying inventory_max items
:- do_take(t,_,_), timestep(t), inventory_max(N), #count{in(O,inventory,t):in(O,inventory,t)} > N.

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

% TEMPORARY OPTIMIZATION HACK don't drop things unless we need to.
:- do_put(t,_,_), inventory_max(N), #count{in(O,inventory,t):in(O,inventory,t)} < N.
:- do_put(t,_,_), not inventory_max(_).

is_action(do_put(t,O,X),t) :- do_put(t,O,X).
on(O,X,t) :- act(do_put(t,O,X),t), instance_of(X,s).  % player puts an object onto a supporting object
in(O,X,t) :- act(do_put(t,O,X),t), instance_of(X,c).  % player puts an object into a container
at(O,X,t) :- act(do_put(t,O,X),t), instance_of(X,r).  % player drops an object to the floor of a room


% ------ CONSUME ------
% drink :: in(f, I) & drinkable(f) -> consumed(f)
%+ drink :: in(f, I) & drinkable(f) & used(slot) -> consumed(f) & free(slot)",
% eat :: in(f, I) & edible(f) -> consumed(f)
%+ eat :: in(f, I) & edible(f) & used(slot) -> consumed(f) & free(slot)",

0 {do_eat(t,F)} 1 :- edible(F,t-1), instance_of(F,f), in(F,inventory,t-1), timestep(t).
0 {do_drink(t,F)} 1 :- drinkable(F,t-1), instance_of(F,f), in(F,inventory,t-1), timestep(t).

% don't consume ingredients that will be needed for the recipe
:- act(do_eat(t,F),t), in_recipe(F), timestep(t).
:- act(do_drink(t,F),t), in_recipe(F), timestep(t).
% don't consume ingredients that might be needed for the recipe
:- act(do_eat(t,F),t), not recipe_read(t), timestep(t).
:- act(do_drink(t,F),t), not recipe_read(t), timestep(t).


is_action(do_eat(t,F),t) :- do_eat(t,F), instance_of(F,f), timestep(t).
is_action(do_drink(t,F),t) :- do_drink(t,F), instance_of(F,f), timestep(t).

consumed(F,t) :- act(do_eat(t,F),t).
consumed(F,t) :- act(do_drink(t,F),t).

consumed(F,t) :- consumed(F,t-1), timestep(t).

% --------------------------------------------------------------------------------

"""

COOKING_RULES = \
"""%---------COOKING_RULES-----------

#program cooking_step(t).
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


% have_prepped_ingredients is True if all required ingredients have been fully prepared and are currently in player's inventory
{have_prepped_ingredients(t)}  :- in_recipe(F), in(F,inventory,t-1), recipe_read(t-1), timestep(t).
:- have_prepped_ingredients(t), in_recipe(F), not in(F,inventory,t-1), timestep(t).
:- have_prepped_ingredients(t), in_recipe(I,F), should_cook(I,V), cookable(F), not cooked_state(F,V,t), timestep(t).
:- have_prepped_ingredients(t), in_recipe(I,F), should_cut(I,V), cuttable(F), not cut_state(F,V,t), timestep(t).
:- have_prepped_ingredients(t), not recipe_read(t-1), timestep(t).

0 { do_make_meal(t) } 1 :- have_prepped_ingredients(t), cooking_location(R, recipe), r(R), at(player,R,t), timestep(t).
:- do_make_meal(t), cooking_location(R, recipe), r(R), not at(player,R,t), timestep(t).

:- do_put(t,_,_), have_prepped_ingredients(t).

is_action(do_make_meal(t),t) :- do_make_meal(t), timestep(t).

in(meal_0,inventory,t) :- act(do_make_meal(t),t), timestep(t).
consumed(F,t) :- act(do_make_meal(t),t), in_recipe(F), timestep(t).

"""
RECIPE_NEED_TO_FIND = \
"""%---------RECIPE_NEED_TO_FIND-----------
need_to_acquire(F,t) :- in_recipe(F), not in(F,inventory,t), not consumed(F,t), not have_acquired(F,t-1). % if t-1, inertia rule applies
:- need_to_acquire(F,t), consumed(F,t).

need_to_acquire(O,t) :- in_recipe(I,F), should_cut(I,V), cuttable(F),
     sharp(O), not cut_state(F,V,t-1), not have_acquired(O,t-1), timestep(t).

{need_to_find(A,t):toaster(A)} 1 :- in_recipe(I,F), should_cook(I,grilled), cookable(F), not cooked_state(F,grilled,t-1), timestep(t).
{need_to_find(A,t):oven(A)} 1 :- in_recipe(I,F), should_cook(I,roasted), cookable(F), not cooked_state(F,roasted,t-1), timestep(t).
{need_to_find(A,t):stove(A)} 1 :- in_recipe(I,F), should_cook(I,fried), cookable(F), not cooked_state(F,fried,t-1), timestep(t).

can_acquire(t) :- {need_acquire(O,t):can_take(O,t)} > 0.
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
%--------------------
#program check(t).
%--------------------

X1=X2 :- act(X1,T), did_act(X2,T), T<t,  query(t).   % explicitly preserve actually chosen actions from previous solving steps

new_acquire(t) :- {first_acquired(O,t):instance_of(O,o)}>0.
new_room(t) :- {first_visited(R,t):r(R)}>0.
newly_opened(t) :- {first_opened(C,t):instance_of(C,c)}>0.

goal1_achieved(t) :- recipe_read(t), not recipe_read(t-1).
goal1_has_been_achieved(t) :- goal1_achieved(t).
goal1_has_been_achieved(t) :- goal1_has_been_achieved(t-1).

goal2_achieved(t) :- consumed(meal_0,t).

solved_all(t) :- goal2_achieved(t).
:- goal2_achieved(t), not goal1_has_been_achieved(t), query(t).  % impose strict sequential ordering: need to read the recipe first


%% Prioritize search (exploring unseen locations) UNTIL WE FIND THE RECIPE
%:- not goal1_has_been_achieved(t), need_to_search(t), not first_visited(_,t), not first_opened(_,t), query(t).
:- not goal1_has_been_achieved(t), need_to_search(t), not new_room(t), not newly_opened(t), query(t).
:- not goal1_has_been_achieved(t), not need_to_search(t), query(t).  % FAIL UNTIL WE FIND the Recipe

% After finding the recipe, if incremental searching or gathering is not required, fail unless we achieve our main goal
:- goal1_has_been_achieved(t-1), not goal2_achieved(t), not need_to_search(t), not need_to_gather(t), query(t).

% Fail unless/until we gather a required item
%:- goal1_has_been_achieved(t-1), not goal2_achieved(t), not need_to_search(t), need_to_gather(t), not first_acquired(_,t), query(t).
:- goal1_has_been_achieved(t-1), not goal2_achieved(t), not need_to_search(t), need_to_gather(t), not new_acquire(t), query(t).
% Fail unless/until we explore a new room or container
%:- goal1_has_been_achieved(t-1), not goal2_achieved(t), need_to_search(t), not need_to_gather(t),
%   not first_visited(_,t), not first_opened(_,t), query(t).
:- goal1_has_been_achieved(t-1), not goal2_achieved(t), need_to_search(t), not need_to_gather(t),
    not new_room(t), not newly_opened(t), query(t).
 
% Allow success upon exploring or gathering, if appropriate
:- goal1_has_been_achieved(t-1), not goal2_achieved(t), need_to_search(t), need_to_gather(t),
         not new_room(t), not newly_opened(t), not new_acquire(t), query(t).
% NOTE (GVS ?unknown why the following don't work vs. new_xxx(t) which do) ----
%      not first_visited(_,t), not first_opened(_,t), not first_acquired(_,t), query(t).

% can_acquire(2) :- t=2, query(t).
 
%------------------------------------------

:- need_to_gather(t), can_acquire(t), {first_visited(R,t):r(R)}>0, query(t). % prefer greedy/1-step acquire over exploring a aew roon
:- need_to_search(t), can_open(t), {first_visited(R,t):r(R)}>0, query(t).    % prefer opening a container over exploring a aew roon
:- need_to_gather(t), can_acquire(t), {first_opened(C,t):instance_of(C,c)}>0, query(t).  % prefer greedy/1-step acquire over opening a new container


%%%%%%%%%%%%%%%%%%%%%%%%%%%%


#show atP/2.
#show goal1_achieved/1.
#show recipe_read/1.
#show goal1_has_been_achieved/1.
#show solved_all/1.

#show can_acquire/1.
#show new_acquire/1.
#show new_room/1.
#show newly_opened/1.

#show first_opened/2.
#show first_visited/2.
#show first_found/2.
#show first_acquired/2.
#show have_prepped_ingredients/1.
#show need_to_gather/1.
#show need_to_search/1.

#show consumed/2.
#show in_recipe/2.
#show in_inventory/2.
#show need_to_find/2.
#show need_to_acquire/2.

%#show have_found/2.
%#show connected/3.
%#show connected/4.
%#show free/3.

% #show contents_unknown/2.
% #show.

"""


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
CONNECTION_REL = ['east_of', 'west_of', 'north_of', 'south_of']
LINK_REL = ['link', 'free']

def is_recipe_fact(fact_name: str, args_str: Optional[str] = None):
    if fact_name.startswith('ingredient'):
        return True
    if args_str and 'ingredient' in args_str:
        return True
    if args_str and 'recipe' in args_str:    # cooking_location(r_0,recipe)
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

def fact_to_asp(fact:Proposition, hfact=None, step:str = '0') -> Tuple[str, bool, bool]:
    """ converts a TextWorld fact to a format appropriate for Answer Set Programming"""
    asp_fact_str, is_part_of_recipe = twfact_to_asp_attrib_state(fact, step)
    if not asp_fact_str:  # not a state attribute, convert to a normal fact
        asp_args = [Variable(twname_to_asp(v.name), v.type) for v in fact.arguments]
        args_str = ', '.join([f"{a.name}" for a in asp_args])
        if is_fluent_fact(fact.name, args_str):
            maybe_timestep = ', '+str(step)
            is_static_fact = False
        else:
            maybe_timestep = ''
            is_static_fact = True
        is_part_of_recipe = is_recipe_fact(fact.name, args_str)   # handles cooking_location, in(ingr,recipe)
        asp_fact_str = f"{fact.name}({args_str}{maybe_timestep})."
    else:
        is_static_fact = is_part_of_recipe   # should_cook(), should_cut() are static, but cook_state(), cut_state() are fluents

    # asp_fact = Proposition(fact.name, asp_args)
    # asp_fact_str = re.sub(r":[^,)]*", '', f"{str(asp_fact)}.")  #remove type annotations (name:type,)
    # if 'ingredient' in asp_fact_str:  #TextWorld quirk: facts refering to 'ingredient_N' are static (define the recipe)
    #     pass   # TODO special-case processing of recipe specifications
    #     # (don't add time step)
    # elif fact.name not in STATIC_FACTS:
    #     asp_fact_str = asp_fact_str.replace(").", f", {step}).")  # add time step to convert initial state facts to fluents
    if hfact:
        asp_fact_str = f"{asp_fact_str} % {str(hfact)}"  # human readable version of the fact (obj names instead of ids)
    return asp_fact_str, is_part_of_recipe, is_static_fact

def twfact_to_asp_attrib_state(fact:Proposition, step:str) -> Tuple[Optional[str], bool]:
    attrib_name, attrib_name_should = is_state_value(fact.name)
    if not attrib_name:
        return None, False  # not an attribute state, and not a recipe step (should_xxx())
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
    return asp_fact_str, is_recipe_entry

# eg_types_to_asp = """
# "types": [
#     "I: I",
#     "P: P",
#     "RECIPE: RECIPE",
#     "c: c -> t",
#     "d: d -> t",
#     "f: f -> o",
#     "ingredient: ingredient -> t",
#     "k: k -> o",
#     "meal: meal -> f",
#     "o: o -> t",
#     "object: object",
#     "oven: oven -> c",
#     "r: r",
#     "s: s -> t",
#     "slot: slot",
#     "stove: stove -> s",
#     "t: t",
#     "toaster: toaster -> c"
#   ],
#  """

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

def _is_a_room(name):
    return name.startswith('r_')

def group_facts_by_room(initial_facts, room_facts):
    first_room = None
    def _add_to_(room_facts, room_name, fact, fact_str):
        # print(f"_add_to_(room_facts, {room_name}, .., '{fact_str}')")
        if room_name in room_facts:
            room_facts[room_name].append(fact_str)
        else:
            room_facts[room_name] = [fact_str]

    where_is = {}
    def _get_room_for(obj_name):
        if _is_a_room(obj_name) or obj_name == 'I':  #obj_name == 'inventory':
            return obj_name
        if obj_name not in where_is:
            return None
        _holder = where_is[obj_name]
        return _get_room_for(_holder)

    for (fact_str, fact) in initial_facts.items():
        #print(fact_str, "|", fact, len(fact.arguments), fact.name)
        if len(fact.arguments) > 1:
            if fact.name in LOCATION_REL:
                if fact.name == 'at':
                    _where = fact.arguments[1].name
                    if not _is_a_room(_where):
                        assert False, f"Expected arg[1] is_a room: {fact_str}"
                    if fact.arguments[0].name == "P":  # initial location of the player
                        assert first_room is None, f"Should be only one, unique at(P,room) - {first_room} {fact}"
                        first_room = _where
                # if fact.name != 'at' or fact.arguments[0].name != "P":  # filter out the player's initial location
                where_is[fact.arguments[0].name] = fact.arguments[1].name
            # elif fact.name == 'free':
            #     room = fact.arguments[0].name
            #     if _is_a_room(room):
            #         _add_to_(room_facts,room,fact,fact_str)
            elif fact.name in CONNECTION_REL or fact.name in LINK_REL:
                if fact.name in CONNECTION_REL:
                    room = fact.arguments[1].name
                else:
                    room = fact.arguments[0].name
                assert _is_a_room(room), f"{fact}"
                _add_to_(room_facts,room,fact,fact_str)

    #print(where_is)
    for obj_name in where_is:
        _where = where_is[obj_name]
    #     if _is_a_room(_where) or 'I' == _where:  # 'inventory' == _where:
    #         print(f"{obj_name}.{where_is[obj_name]}; ", end='')
    # print()
    for (fact_str, fact) in initial_facts.items():
        #print(fact_str, "|", fact, len(fact.arguments), fact.name)
        if len(fact.arguments) > 1 and fact.name in LOCATION_REL:
            _where = _get_room_for(fact.arguments[1].name)
            if _is_a_room(_where):
                # if _where != fact.arguments[1].name: print(fact.arguments[1].name, ":", _where)
                _add_to_(room_facts,_where,fact,fact_str)
            elif _where != 'I':
                print(f"UNEXPECTED: {_where} '{fact_str}'")
    for room in room_facts.keys():
        for fact_str in room_facts[room]:
            assert fact_str in initial_facts, f"[{room}] {room_facts[room]}"
            del initial_facts[fact_str]
    return first_room


def generate_ASP_for_game(game, asp_file_path, hfacts=None):
    if not hfacts:
        _inform7 = Inform7Game(game)
        hfacts = list(map(_inform7.get_human_readable_fact, game.world.facts))

    with open(asp_file_path, "w") as aspfile:
        aspfile.write(INCREMENTAL_SOLVING)
        aspfile.write("% ------- Types -------\n")
        aspfile.write(TYPE_RULES)
        type_infos = types_to_asp(game.kb.types)
        aspfile.write("\n% ------- IS_A -------\n")
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
        aspfile.write("\n\n")
        aspfile.write("\n% ------- Facts -------\n")
        recipe_facts = {}
        room_facts = {}
        initial_fluents = {}
        static_facts = {}
        directions_map = {}
        for fact in game.world.facts:
            if fact.name in CONNECTION_REL: # east_of(R1,R2), north_of, etc...
                r1 = fact.arguments[0].name
                r2 = fact.arguments[1].name
                assert _is_a_room(r1), str(fact)
                assert _is_a_room(r2), str(fact)
                direction = fact.name[0].upper()
                if r2 in directions_map:
                    directions_map[r2][r1] = direction
                else:
                    directions_map[r2] = {r1: direction}
        print(directions_map)
        for fact, hfact in zip(game.world.facts, hfacts):
            fact_str, is_part_of_recipe, is_static_fact = fact_to_asp(fact, hfact, step='t')
            if is_part_of_recipe:
                recipe_facts[fact_str] = fact
            elif is_static_fact and not fact.name in CONNECTION_REL and not fact.name in LINK_REL:
                static_facts[fact_str] = fact
            else:
                if fact.name in CONNECTION_REL:
                    _transition = "_"+fact_str
                    static_facts[_transition] = fact  # privileged knowledge for transitions in partially observable world
                    regex_direction_of = re.compile(r'\(r_(\d)+')      # dir_of(r_1, r_2) => dir_of(unkown, r_2)
                    unknownStr = '(unknown'+fact.name[0].upper()
                    fact_str = regex_direction_of.sub(unknownStr, fact_str)
                    print(fact_str)
                elif fact.name == 'link':
                    regex_link = re.compile(r', r_(\d)+\)')             # link(r_1, d_N, r_2)
                    r1 = fact.arguments[0].name
                    r2 = fact.arguments[2].name
                    unknownStr = f", unknown{directions_map[r1][r2]})"
                    fact_str = regex_link.sub(unknownStr, fact_str)  #  => link(r_1, d_N, unknown)
                    print(fact_str)
                elif fact.name == 'free':
                    #regex_free = re.compile(r', r_(\d)+, ') #.sub(', unknown, ')
                    regex_free = re.compile(r'^free')             # free(r_1, r_2, t)
                    fact_str = regex_free.sub('%free', fact_str)  #  => %free(r_1, r_2, t)
                    print(fact_str)
                initial_fluents[fact_str] = fact
        _initial_room = group_facts_by_room(initial_fluents, room_facts)

        aspfile.write('\n'.join(static_facts.keys()))
        aspfile.write("\n")
        aspfile.write("\n% ------- initial fluents (initialized with t=0) -------\n")
        aspfile.write("#program initial_state(t).\n")
        aspfile.write('\n')
        aspfile.write('\n'.join(initial_fluents.keys()))
        aspfile.write("\n\n")
        aspfile.write("\n")

        aspfile.write("\n% ------- Recipe -------\n")
        aspfile.write("#program recipe(t).\n")

        aspfile.write('\n'.join(recipe_facts.keys()))
        aspfile.write("\n\n")
        aspfile.write("in_recipe(I,F) :- ingredient(I), in(I,recipe), base(F,I), instance_of(F,f).\n")
        aspfile.write("in_recipe(F) :- in_recipe(I,F).\n")
        aspfile.write("\n\n")

        aspfile.write("\n% ------- ROOM fluents -------\n")
        aspfile.write("\n")
        aspfile.write(f"#program initial_room(t).\n")  # special-case the player's initial location
        #aspfile.write(f"at(player, {_initial_room}, t).\n\n")
        for r_fact in room_facts[_initial_room]:
            if r_fact.startswith("at(player,"):
                aspfile.write(r_fact)
        aspfile.write("\n\n")
        for room in sorted(room_facts.keys()):
            aspfile.write(f"#program room_{room}(t).\n")  #TODO: load facts for initial room dynamically
            for r_fact in room_facts[room]:
                if not r_fact.startswith("at(player,"):
                    aspfile.write(r_fact)
                    aspfile.write("\n")
            aspfile.write('\n\n')

        # ---- GAME DYNAMICS
        aspfile.write(EVERY_STEP)
        aspfile.write(EVERY_STEP_MAP_RULES)

        aspfile.write(ACTION_STEP_RULES)
        aspfile.write(GAME_RULES_COMMON)
        aspfile.write(GAME_RULES_NEW)
        aspfile.write(COOKING_RULES)
        aspfile.write(RECIPE_NEED_TO_FIND)


        aspfile.write(CHECK_GOAL_ACHIEVED)
        #aspfile.write(":- movedP(T,R,R1), at(player,R1,T0), timestep(T0), T0<T .  % disallow loops\n")
        # For branch & bound optimization:
        # aspfile.write( #":- not at(player,r_0,maxT).  % end up in the kitchen\n")
        #     "ngoal(T) :- at(player,R,T), r(R), R!=r_0 .  % want to end up in the kitchen (r_0)\n" \
        #     ":- ngoal(maxT).\n  % anti-goal -- fail if goal not achieved"
        # )
        #aspfile.write("_minimize(1,T) :- ngoal(T).\n")

        #aspfile.write("#show.\n")
