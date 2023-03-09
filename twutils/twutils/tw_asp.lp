
% --------------------------------------------------------------------------------
#program check(t).   % redundant declaration (logic is defined below)
#external query(t).
% --------------------------------------------------------------------------------
#program base.
% grounded exactly once, at step 0.

%  NOTE: currently type rules are also in base
% all world vocabulary (object names) are  also defined and grounded at step 0

% ----------------------------------------------------------------------------
% ----------------------         TYPE_RULES            -----------------------
% ----------------------------------------------------------------------------

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

cutting_verb(chop;slice;dice).

cooked_state(needs_cooking;raw;grilled;roasted;fried;burned).
cut_state(uncut;chopped;diced;sliced).

instance_of(X,cooked_state) :- cooked_state(X).
instance_of(X,cut_state) :- cut_state(X).

% additional inheritance links to simplify rules for unknown rooms
unknown(unknownN;unknownS;unknownE;unknownW).  % distinct unknowns for each cardinal direction (N,S,E,W)
instance_of(unknownN;unknownS;unknownE;unknownW, unknown).


% --------------------------------------------------------------------------------
% ---- initialize step 0 values (some fluents that require inertia from step-1)

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


% --------------------------------------------------------------------------------
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


% privileged (fully observable ground truth) knownledge about inventory capacity limit
_inventory_max(N) :- class(slot), N=#count {slot(X):slot(X)}.


% ------------------------------------------------------------------------------
% -------------------        OBSERVATION_STEP          -------------------------
% ------------------------------------------------------------------------------

% ------------------ obs_step(t) t=[0...] ----------------------
#program obs_step(t).  % fluents that are independent of history (don't reference step-1, apply also for step 0)

%_have_found(O,t) :- instance_of(O,thing), _visible_in_room(O,R2,t), room_changed(R,R2,t), timestep(t).
_have_found(O,t) :- instance_of(O,thing), _visible_in_room(O,R,t), at(player,R,t), timestep(t).

_visible_in_room(X,R,t) :- at(X,R,t), r(R).
_visible_in_room(X,R,t) :- on(X,S,t), _visible_in_room(S,R,t), instance_of(S,s), r(R).
_visible_in_room(X,R,t) :- in(X,C,t), _visible_in_room(C,R,t), instance_of(C,c), open(C,t), r(R).


% NOTE: the following MAP_RULES are also evaluated at every observation step
% ------------------------------------------------------------------------------
% -------------       (EVERY_STEP_MAP_RULES)           -------------------------
% ------------------------------------------------------------------------------

% ------- Navigation (evaluated at every step: eventually grounded for all room combinations R1xR2) -------
connected(R1,unknownE,east) :- east_of(unknownE, R1), r(R1).
connected(R1,unknownS,south) :- south_of(unknownS, R1), r(R1).
connected(R1,unknownN,north) :- north_of(unknownN, R1), r(R1).
connected(R1,unknownW,west) :- west_of(unknownW, R1), r(R1).


link(R1,D,R2) :- r(R1), r(R2), d(D), link(R1,D,unknownE), link(R2,D,unknownW).
link(R1,D,R2) :- r(R1), r(R2), d(D), link(R1,D,unknownW), link(R2,D,unknownE).
link(R1,D,R2) :- r(R1), r(R2), d(D), link(R1,D,unknownN), link(R2,D,unknownS).
link(R1,D,R2) :- r(R1), r(R2), d(D), link(R1,D,unknownS), link(R2,D,unknownN).

% ------------------------------------------------------------------------------
% -----------               EVERY_STEP_RULES           -------------------------
% ------------------------------------------------------------------------------

#program predict_step(t).

% is the thing X close enough to the player to interact with
%in_room(X,R,t) :- r(R), r(X), X=R.
in_room(X,R,t) :- at(X,R,t), r(R).
in_room(X,R,t) :- on(X,S,t), in_room(S,R,t), instance_of(S,s), r(R).
in_room(X,R,t) :- in(X,C,t), in_room(C,R,t), instance_of(C,c), open(C,t), r(R).

is_here(X,t) :- at(player,R,t), in_room(X,R,t), r(R).
is_here(R,t) :- at(player,R,t), r(R).
can_take(O,t) :- instance_of(O,o), is_here(O,t). 

need_to_find(X,t) :- need_to_find(X,t-1), not have_found(X,t-1), timestep(t).
need_to_find(X,t) :- need_to_acquire(X,t), not have_found(X,t).
%:- need_to_find(X,t), have_found(X,t-1).

need_to_acquire(X,t) :- need_to_acquire(X,t-1), not have_acquired(X,t-1), timestep(t).
%:- need_to_acquire(X,t), have_acquired(X,t-1).   % if we've gotten it once, deprioritize it
have_acquired(X,t) :- have_acquired(X,t-1).
%MOVED have_acquired(X,t) [-> nearer to have_found(X,t), below]
%have_acquired(X,t) :- in(X,inventory,t), need_to_acquire(X,t-1), timestep(t).

first_acquired(X,t) :- have_acquired(X,t), not have_acquired(X,t-1), timestep(t).

contents_unknown(C,t) :- contents_unknown(C,t-1), timestep(t), closed(C,t).  % not act(do_open(t,C),t).

% newly explored a container (that was previously closed)
first_opened(C,t) :- contents_unknown(C,t-1), not contents_unknown(C,t), timestep(t).

% Alias
%MOVED TO obs_step(t) -- atP(R,t) :- at(player,R,t).                 % alias for player's current location
%in_inventory(O,t) :- in(O,inventory,t).      % alias for object is in inventory at time t

have_examined(O,t) :- act(do_examine(t,O),t), instance_of(O,thing), timestep(t).
have_examined(R,t) :- act(do_look(t,R),t), r(R), timestep(t).

% inertia
have_examined(X,t) :- have_examined(X,t-1), timestep(t).  %, not act(do_examine(t,X),t), not act(do_look(t,X),t).

recipe_read(t) :- have_examined(o_0,t).   % o_0 is always the RECIPE
% recipe_read(t) :- act(do_examine(t,o_0),t), timestep(t).   % o_0 is always the RECIPE
% recipe_read(t) :- recipe_read(t-1), timestep(t).

% --- inertia: doors and containers don't change state unless player acts on them
open(X,t) :- is_openable(X), open(X,t-1), not act(do_close(t,X),t).
open(X,t) :- is_openable(X), act(do_open(t,X),t).
closed(X,t) :- closed(X,t-1), not act(do_open(t,X),t).
locked(X,t) :- locked(X,t-1), not act(do_unlock(t,X,_),t).
% ------ CONSTRAINTS ------
:- open(X,t), closed(X,t).  %[,is_openable(X).]    % any door or container can be either open or closed but not both
:- locked(X,t), open(X,t).  %[,is_lockable(X).]    % can't be both locked and open at the same time

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

%GVS 22-02-2023
:- cooked(X,t-1), act(do_cook(t,X,_), t).    % explicitly prevent burning already cooked food.

cooked_state(X,grilled,t) :- act(do_cook(t,X,A), t), instance_of(A,toaster), not cooked(X,t-1).  % cooking with a BBQ or grill or toaster
cooked_state(X,fried,t) :- act(do_cook(t,X,A), t), instance_of(A,stove), not cooked(X,t-1).   % cooking on a stove => frying
cooked_state(X,roasted,t) :- act(do_cook(t,X,A), t), instance_of(A,oven), not cooked(X,t-1).   % cooking in an oven => roasting
cooked_state(X,burned,t) :- cooked(X,t-1), act(do_cook(t,X,_), t).   % cooking something twice causes it to burn

inedible(X,t) :- inedible(X,t-1), not act(do_cook(t,X,_),t).
edible(X,t) :- edible(X,t-1), not act(do_cook(t,X,_),t).
%edible(X,t) :- cooked_state(X,needs_cooking,t-1), inedible(X,t-1), not cooked(X,t-1), cooked(X,t). % cooking => transition from inedible to edible
edible(X,t) :- cooked(X,t), cooked_state(X,V,t), V!=burned. % cooking => transition from inedible to edible


%count_inventory(N,t) :- timestep(t), N=#sum{1,in(O,inventory,t):in(O,inventory,t)}.  % #sum of 1, = same as #count{}
inventory_count(t,N) :- N = #count{in(O,inventory,t):in(O,inventory,t)}.

% (unobserved) ground truth information about whether the inventory is already full
%inventory_is_full(t) :- _inventory_max(N), #count{in(O,inventory,t):in(O,inventory,t)} >= N.
_inventory_is_full(t) :- _inventory_max(M), inventory_count(t,N), N >= M.

% or based on the agent's belief state (if have observed an event that reveals inventory capacity)
inventory_is_full(t) :- inventory_capacity(M,t), inventory_count(t,N), N >= M.


% ---- inertia: objects don't move unless moved by the player
did_take(X,Y,t) :- act(do_take(t,X,Y), t), not _inventory_is_full(t-1).
% by attempting and failing to take something, we learn inventory capacity limit
inventory_capacity(N,t) :- act(do_take(t,X,Y), t), _inventory_is_full(t-1), inventory_count(t-1,N).
inventory_capacity(N,t) :- inventory_capacity(N,t-1).


at(X,R,t) :- at(X,R,t-1), r(R), instance_of(X,thing), not did_take(X,R,t).
on(X,S,t) :- on(X,S,t-1), instance_of(S,s), not did_take(X,S,t).
in(X,C,t) :- in(X,C,t-1), instance_of(C,c), not did_take(X,C,t).

in(X,inventory,t) :- in(X,inventory,t-1), not act(do_put(t,X,_),t), not consumed(X,t).

in(O,inventory,t) :- did_take(O,_,t).  % if player takes an object, it moves to the inventory

on(O,X,t) :- act(do_put(t,O,X),t), instance_of(X,s).  % player puts an object onto a supporting object
in(O,X,t) :- act(do_put(t,O,X),t), instance_of(X,c).  % player puts an object into a container
at(O,X,t) :- act(do_put(t,O,X),t), instance_of(X,r).  % player drops an object to the floor of a room


% ------ CONSTRAINTS ------
:- cuttable(X), {cut_state(X,V,t):instance_of(V,cut_state) } > 1.   % disjoint set of attribute values for cuttable items

% --- inertia: cuttable items change state only if the player acts on them
cut_state(X,uncut,t) :- cut_state(X,uncut,t-1), not act(do_cut(t,_,X,_),t).
cut_state(X,chopped,t) :- cut_state(X,uncut,t-1), act(do_cut(t,chop,X,_),t).
cut_state(X,diced,t) :- cut_state(X,uncut,t-1), act(do_cut(t,dice,X,_),t).
cut_state(X,sliced,t) :- cut_state(X,uncut,t-1), act(do_cut(t,slice,X,_),t).

% cut-up items remain cut-up, and can't be cut up any further
cut_state(X,V,t) :- cut_state(X,V,t-1), V != uncut.


consumed(F,t) :- act(do_eat(t,F),t).
consumed(F,t) :- act(do_drink(t,F),t).

consumed(F,t) :- consumed(F,t-1), timestep(t).


% ------------------ predict_step(t) t=[0...] ----------------------
#program predict_step(t).    % applied at each timestep >=0

% define fluents that determine whether player can move from room R0 to R1
%%%%%%%% TODO: ? add NSEW to free(R0,R1,NSEW,t)

free(R0,R1,t) :- r(R0), d(D), link(R0,D,R1), open(D,t). %, r(R1)
not free(R0,R1,t) :- r(R0), d(D), link(R0,D,R1), not open(D,t). %, r(R1)

%free(R0,R1,t) :- r(R0), connected(R0,R1), not link(R0,_,R1).  %, r(R1), % if there is no door, exit is always traversible.
free(R0,R1,t) :- r(R0), connected(R0,R1,NSEW), direction(NSEW), not link(R0,_,R1).  %, r(R1), % if there is no door, exit is always traversible.
free(R0,R1,t) :- r(R0), connected(R0,R1,NSEW,t), not door_direction(R0,_,NSEW).  %, r(R1), % if there is no door, exit is always traversible.
% inertia
%free(R0,R1,t) :- free(R0,R1,t-1), not link(R0,_,R1).  % no door -> can never become closed/unpassable


connected(R1,R2,east,t) :- act(do_moveP(t,R1,unknownE,east),t), room_changed(R1,R2,t), r(R2), R1!=R2.
connected(R1,R2,south,t) :- act(do_moveP(t,R1,unknownS,south),t), room_changed(R1,R2,t), r(R2), R1!=R2.
connected(R1,R2,north,t) :- act(do_moveP(t,R1,unknownN,north),t), room_changed(R1,R2,t), r(R2), R1!=R2.
connected(R1,R2,west,t) :- act(do_moveP(t,R1,unknownW,west),t), room_changed(R1,R2,t), r(R2), R1!=R2.

% assume that all doors/exits can be traversed in both directions
% assume that all doors/exits can be traversed in both directions
connected(R1,R2,east,t) :- connected(R2,R1,west,t), r(R1).
connected(R1,R2,west,t) :- connected(R2,R1,east,t), r(R1).
connected(R1,R2,south,t) :- connected(R2,R1,north,t), r(R1).
connected(R1,R2,north,t) :- connected(R2,R1,south,t), r(R1).

% inertia
connected(R1,R2,NSEW,t) :- connected(R1,R2,NSEW,t-1),r(R1),r(R2),direction(NSEW). % once connected -> always connected

% --- inertia: player stays in the current room unless acts to move to another
  % stay in the current room unless current action is do_moveP
%at(player,R0,t) :- at(player,R0,t-1), r(R0), {act(do_moveP(t,R0,R,NSEW),t):r(R),direction(NSEW)}=0. %, T<=maxT.
%at(player,R0,t) :- at(player,R0,t-1), r(R0), not act(do_moveP(t,R0,_,NSEW),t):direction(NSEW). %, T<=maxT.
at(player,R,t) :- at(player,R,t-1), not room_changed(R,_,t).

  % player moved at time t, from previous room R0 to new room R
%room_changed(R0,R,t) :- act(do_moveP(t,R0,_,NSEW),t), at(player,R0,t-1), r(R0), _connected(R0,R,NSEW), direction(NSEW). %, R!=R0.
at(player,R,t) :- room_changed(R0,R,t).

atP(R,t) :- at(player,R,t), r(R).   % Alias for player's initial position"
in_inventory(O,t) :- in(O,inventory,t).      % alias for object is in inventory at time t


% player has visited room R
have_found(R,t) :- r(R), room_changed(R0,R,t).
first_found(X,t) :- have_found(X,t), not have_found(X,t-1).
first_visited(R,t) :- r(R), first_found(R,t).

%inertia: once we've found something, it stays found forever
% (because the player is the only agent in a deterministic world, with perfect memory)
have_found(X,t) :- have_found(X,t-1).

%SIMPLIFIED: have_found(O,t) :- at(player,R,t), r(R), instance_of(O,o), in(O,inventory,t), timestep(t).
have_found(O,t) :- in(O,inventory,t), timestep(t).
have_acquired(O,t) :- in(O,inventory,t), timestep(t).
have_found(O,t) :- _have_found(O,t-1).

% can see a thing if player is in the same room as the thing
%have_found(O,t) :- instance_of(O,thing), is_here(O,t), timestep(t).
% can see an object that is on a support if player is in the same room as the suppport
%have_found(O,t) :- at(player,R,t), r(R), at(S,R,t), on(O,S,t), timestep(t).
% can see an object that's in a container if the container is open and player is in the same room
%have_found(O,t) :- at(player,R,t), r(R), instance_of(O,o), instance_of(C,c), at(C,R,t), in(O,C,t), open(C,t), timestep(t).

need_to_search(t) :- {not have_found(X,t):need_to_find(X,t)}>0.

need_to_gather(t) :- {need_to_acquire(X,t):need_to_acquire(X,t)}>0.


% ------------------------------------------------------------------------------
% ----------------            PREVIOUS ACTIONS            ----------------------
% ------------------------------------------------------------------------------
#program prev_action(a,t).
%#external did_act(a,t).
% did_act(a,t).
:- act(X,t), X != a.   % past actions cannot be changed

% ------------------------------------------------------------------------------
% ----------------            ACTION_STEP_RULES           ----------------------
% ------------------------------------------------------------------------------

% ------------------ step(t) t=[1...] ----------------------
#program step(t).    % applied at each timestep >=1

% Generate
timestep(t).

{act(X,t):is_action(X,t)} = 1 :- timestep(t). % player must choose exactly one action at each time step.

{at(player,R,t):r(R)} = 1 :- timestep(t).   % player is in exactly one room at any given time
% NOTE/IMPORTANT - THE FOLLOWING IS NOT THE SAME as prev line, DOES NOT WORK CORRECTLY:
%  {at(player,R,t)} = 1 :- r(R), timestep(t).
% NOTE - THE FOLLOWING ALSO DOESN'T WORK (expands too many do_moveP ground instances at final timestep:
%  {at(player,R,t)} = 1 :- r(R), at(player,R,t), timestep(t).


% ------------------------------------------------------------------------------
% -------------------            GAME_RULES_COMMON           -------------------
% ------------------------------------------------------------------------------

% ------ LOOK ------
% inventory :: $at(P, r) -> 
% look :: $at(P, r) -> 
% ------ LOOK AT: EXAMINE an object ------
% examine/I :: $at(o, I) -> 
% examine/c :: $at(P, r) & $at(c, r) & $open(c) & $in(o, c) -> 
% examine/s :: $at(P, r) & $at(s, r) & $on(o, s) -> 
% examine/t :: $at(P, r) & $at(t, r) -> 


0 {do_look(t,R):r(R),at(player,R,t)} 1 :- at(player,R,t), r(R), timestep(t).

0 {do_examine(t,O):is_here(O,t),instance_of(O,thing)} 1 :- is_here(O,t), instance_of(O,thing), timestep(t).
0 {do_examine(t,O):in(O,inventory,t)} 1 :- at(player,R,t), r(R), instance_of(O,o), in(O,inventory,t), timestep(t).

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


% ------ GO ------
% go/east :: at(P, r) & $west_of(r, r') & $free(r, r') & $free(r', r) -> at(P, r')
% go/north :: at(P, r) & $north_of(r', r) & $free(r, r') & $free(r', r) -> at(P, r')
% go/south :: at(P, r) & $north_of(r, r') & $free(r, r') & $free(r', r) -> at(P, r')
% go/west :: at(P, r) & $west_of(r', r) & $free(r, r') & $free(r', r) -> at(P, r')


% Test constraints
%:- at(player,R0,t-1), at(player,R,t), r(R0), r(R), R!=R0, not free(R,R0,t-1).

 % can move to a connected room, if not blocked by a closed door
0 {do_moveP(t,R0,R,NSEW):free(R0,R,t),connected(R0,R,NSEW),direction(NSEW)} 1 :-
                         free(R0,R,t),connected(R0,R,NSEW),direction(NSEW), at(player,R0,t-1), r(R0). %
0 {do_moveP(t,R0,R,NSEW):free(R0,R,t),connected(R0,R,NSEW,t-1),direction(NSEW),at(player,R0,t-1)} 1 :-
                         free(R0,R,t),connected(R0,R,NSEW,t-1),direction(NSEW),at(player,R0,t-1), r(R0). %

room_changed(R0,R,t) :- act(do_moveP(t,R0,_,NSEW),t), at(player,R0,t-1), r(R0), _connected(R0,R,NSEW), direction(NSEW). %, R!=R0.

% Test constraints
:- do_moveP(t,R0,U,NSEW),unknown(U),direction(NSEW),r(R0),timestep(t),not free(R0,U,t).  % can't go that way: not a valid action
:- do_moveP(t,R0,R,NSEW),direction(NSEW),r(R0),r(R),timestep(t),not free(R0,R,t).  % can't go that way: not a valid action
:- do_moveP(t,R0,U,NSEW),unknown(U),direction(NSEW),r(R0),timestep(t),room_changed(R0,R2,t),free(R0,R2,t-1).  % if dest room is known, use it explicitly

is_action(do_moveP(t,R1,R2,NSEW), t) :- do_moveP(t,R1,R2,NSEW). %, r(R1), r(R2), direction(NSEW).

%++++++  DEBUGGING
%do_look(t,R) :- first_visited(R,t-1).
%%is_action(do_look(t,R)) :- room_changed(R0,R,t-1).
%act(do_look(t,R), t) :- first_visited(R,t-1).
%------  DEBUGGING


% ------ OPEN/CLOSE UNLOCK/LOCK ------
% close/c :: $at(P, r) & $at(c, r) & open(c) -> closed(c)
% close/d :: $at(P, r) & $link(r, d, r') & $link(r', d, r) & open(d) & free(r, r') & free(r', r) -> closed(d)
% open/c :: $at(P, r) & $at(c, r) & closed(c) -> open(c)
% open/d :: $at(P, r) & $link(r, d, r') & $link(r', d, r) & closed(d) -> open(d) & free(r, r') & free(r', r)
% lock/c :: $at(P, r) & $at(c, r) & $in(k, I) & $match(k, c) & closed(c) -> locked(c)
% lock/d :: $at(P, r) & $link(r, d, r') & $link(r', d, r) & $in(k, I) & $match(k, d) & closed(d) -> locked(d)
% unlock/c :: $at(P, r) & $at(c, r) & $in(k, I) & $match(k, c) & locked(c) -> closed(c)
% unlock/d :: $at(P, r) & $link(r, d, r') & $link(r', d, r) & $in(k, I) & $match(k, d) & locked(d) -> closed(d)


%has_door(R,D) :- r(R), d(D), link(R,D,_).
has_door(R,D) :- r(R), d(D), direction(NSEW), link(R,D,R2), connected(R,R2,NSEW).
door_direction(R,D,NSEW) :- r(R), d(D), direction(NSEW), link(R,D,R2), connected(R,R2,NSEW).

% can open a closed but unlocked door
0 {do_open(t,D):d(D),closed(D,t-1),at(player,R0,t),has_door(R0,D)} 1 :- at(player,R0,t), has_door(R0,D), d(D), closed(D,t-1), not locked(D,t-1). % R1 -- might be unknown: = not a room
% can open a closed but unlocked container
0 {do_open(t,C):instance_of(C,c),closed(C,t-1),is_here(C,t)} 1 :- is_here(C,t), instance_of(C,c), closed(C,t-1), not locked(C,t-1).
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


0 {do_cook(t,X,A):cookable(X),instance_of(A,cooker)} 1 :- at(player,R,t-1), r(R), cookable(X), instance_of(A,cooker), in(X,inventory,t-1), at(A,R,t-1).
% Test constraints
:- do_cook(t,X,A), at(player,R,t), at(A,R2,t), R != R2. % can't cook using an appliance that isn't in the current room

is_action(do_cook(t,X,A), t) :- do_cook(t,X,A).

% ------ CUT ------
% chop :: $in(f, I) & $in(o, I) & $sharp(o) & uncut(f) -> chopped(f)
% dice :: $in(f, I) & $in(o, I) & $sharp(o) & uncut(f) -> diced(f)
% slice :: $in(f, I) & $in(o, I) & $sharp(o) & uncut(f) -> sliced(f)

% can chop, slice or dice cuttable ingredients that are in player's inventory if also have a knife (a sharp object), 
0 {do_cut(t,V,F,O):cutting_verb(V),cuttable(F),sharp(O),in(O,inventory,t-1),in(F,inventory,t-1) } 1 :- 
    cuttable(F), cut_state(F,uncut,t-1), in(F,inventory,t-1), sharp(O), in(O,inventory,t-1). %, not cooked(F,t-1).
%:- do_cut(t,_,F,_), cooked(F,t).       % can't cut up an ingredient that has already been cooked (in TextWorld)

:- do_cut(t,_,F,O), not cut_state(F,uncut,t-1).  % can't cut up something that's already cut up
:- do_cut(t,_,F,O), not sharp(O).      % can't cut up something with an unsharp instrument

is_action(do_cut(t,V,F,O), t) :- do_cut(t,V,F,O).

% ------------------------------------------------------------------------------
% -----------------             GAME_RULES_NEW            ----------------------
% ------------------------------------------------------------------------------

% ------ TAKE ------
%- take :: $at(P, r) & at(o, r) -> in(o, I)
%+ take :: $at(P, r) & at(o, r) & free(slot) -> in(o, I) & used(slot)",
%- take/c :: $at(P, r) & $at(c, r) & $open(c) & in(o, c) -> in(o, I)
%+ take/c :: $at(P, r) & $at(c, r) & $open(c) & in(o, c) & free(slot) -> in(o, I) & used(slot)",
%- take/s :: $at(P, r) & $at(s, r) & on(o, s) -> in(o, I)
%+ take/s :: $at(P, r) & $at(s, r) & on(o, s) & free(slot) -> in(o, I) & used(slot)",


% -- take/c :: can take an object that's in a container if the container is open and player is in the same room
0 {do_take(t,O,C):in(O,C,t-1)} 1 :- at(player,R,t-1), r(R), instance_of(O,o), instance_of(C,c), at(C,R,t-1), in(O,C,t-1), open(C,t-1), timestep(t).
% -- take/s :: can take an object that's on a support if player is in the same room as the suppport
0 {do_take(t,O,S):on(O,S,t-1)} 1 :- at(player,R,t-1), r(R), instance_of(O,o), instance_of(S,s), at(S,R,t-1), on(O,S,t-1), timestep(t).

% -- take :: can take an object (a portable thing) if player is in the same room and it is on the floor
0 {do_take(t,O,R):at(O,R,t-1),at(player,R,t-1)} 1 :- at(player,R,t-1), r(R), instance_of(O,o), at(O,R,t-1), timestep(t).


% don't try to pick up anything if already carrying inventory_max items (if inventory_max is known)
:- do_take(t,_,_), timestep(t), inventory_is_full(t-1).

is_action(do_take(t,O,X),t) :- do_take(t,O,X).


% ------ DROP/PUT ------
% put :: $at(P, r) & $at(s, r) & in(o, I) -> on(o, s)
%+ put :: $at(P, r) & $at(s, r) & in(o, I) & used(slot) -> on(o, s) & free(slot)",
% drop :: $at(P, r) & in(o, I) -> at(o, r)
%+ drop :: $at(P, r) & in(o, I) & used(slot) -> at(o, r) & free(slot)",
% insert :: $at(P, r) & $at(c, r) & $open(c) & in(o, I) -> in(o, c)
%+ insert :: $at(P, r) & $at(c, r) & $open(c) & in(o, I) & used(slot) -> in(o, c) & free(slot)",

% insert :: can put an object into a container if the player has the object, container is open and player is in the same room as container
0 {do_put(t,O,C):in(O,inventory,t-1),instance_of(C,c)} 1 :- at(player,R,t-1), r(R), instance_of(O,o), instance_of(C,c), at(C,R,t-1), in(O,inventory,t-1), open(C,t-1), timestep(t).
% put :: can put an object onto a support if player has the object and is in the same room as the suppport
0 {do_put(t,O,S):in(O,inventory,t-1),instance_of(S,s)} 1 :- at(player,R,t-1), r(R), instance_of(O,o), s(S), at(S,R,t-1), in(O,inventory,t-1), timestep(t).
% drop :: can drop an object on the floor of a room if player is in the room and has the object
0 {do_put(t,O,R):in(O,inventory,t-1),at(player,R,t-1)} 1 :- at(player,R,t-1), r(R), instance_of(O,o), in(O,inventory,t-1), timestep(t).

% TEMPORARY OPTIMIZATION HACK don't drop things unless we need to.
%:- do_put(t,_,_), inventory_max(N), #count{in(O,inventory,t):in(O,inventory,t)} < N.
%:- do_put(t,_,_), not inventory_max(_).

is_action(do_put(t,O,X),t) :- do_put(t,O,X).

% ------ CONSUME ------
% drink :: in(f, I) & drinkable(f) -> consumed(f)
%+ drink :: in(f, I) & drinkable(f) & used(slot) -> consumed(f) & free(slot)",
% eat :: in(f, I) & edible(f) -> consumed(f)
%+ eat :: in(f, I) & edible(f) & used(slot) -> consumed(f) & free(slot)",

0 {do_eat(t,F):edible(F,t-1),in(F,inventory,t-1)} 1 :- edible(F,t-1), instance_of(F,f), in(F,inventory,t-1), timestep(t).
0 {do_drink(t,F):drinkable(F,t-1),in(F,inventory,t-1)} 1 :- drinkable(F,t-1), instance_of(F,f), in(F,inventory,t-1), timestep(t).

% don't consume ingredients that will be needed for the recipe
:- act(do_eat(t,F),t), in_recipe(F), timestep(t).
:- act(do_drink(t,F),t), in_recipe(F), timestep(t).
% don't consume ingredients that might be needed for the recipe
:- act(do_eat(t,F),t), not recipe_read(t), timestep(t).
:- act(do_drink(t,F),t), not recipe_read(t), timestep(t).


is_action(do_eat(t,F),t) :- do_eat(t,F), instance_of(F,f), timestep(t).
is_action(do_drink(t,F),t) :- do_drink(t,F), instance_of(F,f), timestep(t).

% ------------------------------------------------------------------------------
% ------------------               COOKING_RULES           ---------------------
% ------------------------------------------------------------------------------

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
not_available_ingredient(F,t) :- in_recipe(F), not in(F,inventory,t-1), timestep(t).
not_cooked_ingredient(F,t) :- in_recipe(I,F), should_cook(I,V), cookable(F), not cooked_state(F,V,t), timestep(t).
not_cut_ingredient(F,t) :- in_recipe(I,F), should_cut(I,V), cuttable(F), not cut_state(F,V,t), timestep(t).
:- have_prepped_ingredients(t), not_available_ingredient(F,t).
:- have_prepped_ingredients(t), not_cooked_ingredient(F,t).
:- have_prepped_ingredients(t), not_cut_ingredient(F,t).
:- have_prepped_ingredients(t), not recipe_read(t-1), timestep(t).

0 { do_make_meal(t) } 1 :- have_prepped_ingredients(t), cooking_location(R,recipe), r(R), at(player,R,t), timestep(t).
:- do_make_meal(t), cooking_location(R, recipe), r(R), not at(player,R,t), timestep(t).

:- do_put(t,_,_), have_prepped_ingredients(t).

is_action(do_make_meal(t),t) :- do_make_meal(t), timestep(t).

in(meal_0,inventory,t) :- act(do_make_meal(t),t), timestep(t).
consumed(F,t) :- act(do_make_meal(t),t), in_recipe(F), timestep(t).

% ------------------------------------------------------------------------------
% ----------------             RECIPE_NEED_TO_FIND            ------------------
% ------------------------------------------------------------------------------

%---------RECIPE_NEED_TO_FIND-----------
need_to_acquire(F,t) :- in_recipe(F), not in(F,inventory,t), not consumed(F,t), not have_acquired(F,t). % if t-1, inertia rule applies
:- need_to_acquire(F,t), consumed(F,t).

need_to_acquire(O,t) :- in_recipe(I,F), should_cut(I,V), cuttable(F),
     sharp(O), not cut_state(F,V,t), not have_acquired(O,t), timestep(t).

%{need_to_find(A,t):toaster(A)} 1 :- in_recipe(I,F), should_cook(I,grilled), cookable(F), not cooked_state(F,grilled,t-1), timestep(t).
%{need_to_find(A,t):oven(A)} 1 :- in_recipe(I,F), should_cook(I,roasted), cookable(F), not cooked_state(F,roasted,t-1), timestep(t).
%{need_to_find(A,t):stove(A)} 1 :- in_recipe(I,F), should_cook(I,fried), cookable(F), not cooked_state(F,fried,t-1), timestep(t).

%{need_to_find(A,t):toaster(A),not have_found(A,t-1)} 1 :-
 need_to_find(A,t) :-
    in_recipe(I,F), should_cook(I,grilled), cookable(F), not cooked_state(F,grilled,t-1), toaster(A), not have_found(A,t), timestep(t).
%{need_to_find(A,t):oven(A),not have_found(A,t-1)} 1 :-
need_to_find(A,t) :-
     in_recipe(I,F), should_cook(I,roasted), cookable(F), not cooked_state(F,roasted,t-1), oven(A), not have_found(A,t), timestep(t).
%{need_to_find(A,t):stove(A),not have_found(A,t-1)} 1 :-
need_to_find(A,t) :-
     in_recipe(I,F), should_cook(I,fried), cookable(F), not cooked_state(F,fried,t-1), stove(A), not have_found(A,t), timestep(t).

%can_acquire(t) :- {need_to_acquire(O,t):can_take(O,t)} > 0.

% ------------------------------------------------------------------------------
% -----------------           CHECK_GOAL_ACHIEVED            -------------------
% ------------------------------------------------------------------------------

%--------------------
#program check(t).
%--------------------
% explicitly preserve actually chosen actions from previous solving steps
%SX1=X2 :- act(X1,T), did_act(X2,T), T<t,  query(t).

can_acquire(t) :- {need_to_acquire(O,t-1):can_take(O,t-1)} > 0, query(t).
can_open(t) :- {contents_unknown(C,t-1):is_here(C,t-1),instance_of(C,c)} > 0, query(t).

inventory_max_found(t) :- inventory_capacity(M,t), not inventory_capacity(M,t-1), query(t).
new_acquire(t) :- {first_acquired(O,t):need_to_acquire(O,t-1)}>0.
new_room(t) :- {first_visited(R,t):r(R)}>0.
newly_opened(t) :- {first_opened(C,t):instance_of(C,c)}>0.

goal1_achieved(t) :- recipe_read(t), not recipe_read(t-1).
goal1_has_been_achieved(t) :- goal1_achieved(t).
goal1_has_been_achieved(t) :- goal1_has_been_achieved(t-1).

goal2_achieved(t) :- consumed(meal_0,t).

solved_all(t) :- goal2_achieved(t).

% ------------------------------------------------------------------------------
% Constraints (implement prioritization of subgoals relative to each other)

% if we don't need to gather stuff, inventory capacity is irrelevant
:- not need_to_gather(t), inventory_max_found(t), query(t).

% prefer greedy/1-step acquire over exploring a aew room
:- can_acquire(t), new_room(t), query(t).

% prefer greedy/1-step acquire over opening a new container
:- can_acquire(t), newly_opened(t), query(t).

% prefer opening a container over exploring a aew room
:- need_to_search(t), can_open(t), new_room(t), query(t).

% enforce strict sequential ordering: need to read the recipe before preparing the meal
:- goal2_achieved(t), not goal1_has_been_achieved(t), query(t).

% if we know where the recipe is, FAIL UNTIL WE READ the Recipe
:- not need_to_search(t), not goal1_has_been_achieved(t), t>0, query(t).

%% UNTIL WE FIND THE RECIPE: allow solver success for search (exploring unseen locations)
:- not goal1_has_been_achieved(t),
      need_to_search(t),
          not new_room(t), not newly_opened(t), not inventory_max_found(t), query(t).

% ****** After finding the recipe: ******

% if searching or gathering is not required, fail until we achieve our main goal
:- goal1_has_been_achieved(t-1), not goal2_achieved(t),
     not need_to_search(t), not need_to_gather(t), query(t).

% Fail unless/until we gather a required item
:- goal1_has_been_achieved(t-1), not goal2_achieved(t),
     not need_to_search(t), need_to_gather(t),
          not new_acquire(t), not inventory_max_found(t), query(t).
% Fail unless/until we explore a new room or container
:- goal1_has_been_achieved(t-1), not goal2_achieved(t),
     need_to_search(t), not need_to_gather(t),
           not new_room(t), not newly_opened(t), query(t).
 
% Allow success upon exploring or gathering, if appropriate
:- goal1_has_been_achieved(t-1), not goal2_achieved(t),
      need_to_search(t), need_to_gather(t),
         not new_room(t), not newly_opened(t),
         not new_acquire(t), not inventory_max_found(t), query(t).
% NOTE (GVS ?unknown why the following don't work vs. new_xxx(t) which do) ----
%      not first_visited(_,t), not first_opened(_,t), not first_acquired(_,t), query(t).


%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#show did_act/2.
#show atP/2.
#show A: act(A,T),timestep(T).
#show inventory_capacity/2.
#show inventory_max_found/1.
#show inventory_is_full/1.
#show _inventory_is_full/1.

%#show goal1_achieved/1.
%#show recipe_read/1.
%#show goal1_has_been_achieved/1.
%#show solved_all/1.

%#show can_acquire/1.
%#show new_acquire/1.
%#show new_room/1.
%#show newly_opened/1.

%#show first_opened/2.
%#show first_visited/2.
%#show first_found/2.
%#show first_acquired/2.
%#show have_prepped_ingredients/1.
%#show need_to_gather/1.
%#show need_to_search/1.

%#show consumed/2.
%#show in_recipe/2.
%#show in_inventory/2.
%#show need_to_find/2.
%#show need_to_acquire/2.

%#show have_found/2.
%#show is_here/2.

%#show connected/3.
%#show connected/4.
%#show free/3.

% #show contents_unknown/2.

% #show X : holds(X).  % Example: how to show terms of predicates in the answer set

#show.
