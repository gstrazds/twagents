
% --------------------------------------------------------------------------------
#program check(t).   % redundant declaration (logic is defined below)
#external query(t).
% --------------------------------------------------------------------------------
#program base.
% grounded exactly once, at step 0.

%  NOTE: currently type rules are also in base
% all world vocabulary (object names) are  also defined and grounded at step 0

% ======= Types ======
% ------- IS_A -------
instance_of(X,inventory) :- inventory(X).
instance_of(X,player) :- player(X).
instance_of(X,direction) :- direction(X).
instance_of(X,thing) :- thing(X).   % base class for things that aren't rooms/locations/directions
instance_of(X,c) :- c(X).  % container:  in(o, c)
instance_of(X,d) :- d(X).  % door
instance_of(X,k) :- k(X).  % key
instance_of(X,o) :- o(X).  % portable object
instance_of(X,r) :- r(X).  % room
instance_of(X,s) :- s(X).  % support: on(o, s)
instance_of(gpos(X,Y),gridpos) :- gpos(X,Y).
instance_of(X,box) :- box(X).  % box
instance_of(X,ball) :- ball(X). % ball
instance_of(X,tile) :- tile(X).  % floor tile
instance_of(X,wall) :- wall(X). % wall
instance_of(X,lava) :- lava(X). % lava
instance_of(X,goal) :- goal(X). % goal

subclass_of(c,thing).
subclass_of(d,thing).
subclass_of(o,thing).
subclass_of(s,thing).
subclass_of(k,o).     % keys can be carried
subclass_of(ball,o).  % balls can be carried
subclass_of(box,o).   % boxes can be carried
subclass_of(box,c).   % boxes can be containers
subclass_of(lava,tile).
subclass_of(wall,tile).
subclass_of(goal,tile).

%instance_of(X,f) :- f(X). % food
%instance_of(X,recipe) :- recipe(X).
%instance_of(X,ingredient) :- ingredient(X).
%instance_of(X,meal) :- meal(X).
%instance_of(X,oven) :- oven(X).
%instance_of(X,stove) :- stove(X).
%instance_of(X,toaster) :- toaster(X).
%subclass_of(f,o).
%subclass_of(ingredient,thing).
%subclass_of(meal,f).
%subclass_of(stove,s).
%subclass_of(toaster,c).

% ----------------------------------------------------------------------------
% ----------------------         TYPE_RULES            -----------------------
% ----------------------------------------------------------------------------

subclass_of(A,C) :- subclass_of(A,B), subclass_of(B,C).  % subclass relationship is transitive
instance_of(I,B) :- instance_of(I,A), subclass_of(A,B).  % an instance of a subclass is also an instance of the superclass
class(C) :- instance_of(X,C).  % instance_of relationship implicity defines classes
class(S) :- subclass_of(S,C).  % every subclass is also a class
class(S) :- subclass_of(S,C).  % redundant [with instance-of-superclass rule, above]

% additional inheritance links to consolidate 3 rules for cooking with appliances
%class(cooker).
%subclass_of(oven,cooker).
%subclass_of(stove,cooker).
%subclass_of(toaster,cooker).

% additional inheritance links to simplify rules for attributes related to cooked_state and cut_state
class(attribute_value).
subclass_of(cooked_state, attribute_value).
subclass_of(cut_state, attribute_value).

%{is_openable(X); is_lockable(X)}=2 :- instance_of(X,d). % doors are potentially openable and lockable
is_openable(X) :- instance_of(X,d).  % doors are potentially openable
is_lockable(X) :- instance_of(X,d).  % doors are potentially lockable
is_openable(X) :- instance_of(X,c).  % containers are potentially openable
is_lockable(X) :- instance_of(X,c), not instance_of(X,box). % , not instance_of(X,oven) % some containers might be lockable

% action vocabulary
timestep(0). % incremental solving will define timestep(t) for t >= 1...

direction(east;west;north;south).
unknown(unknown).

stepmove(forward;left;right).  % player can move forward or turn right or left

turn_result(east,left,north).  % facing east + turn left -> facing north
turn_result(north,left,west).  % facing north + turn left -> facing west
turn_result(west,left,south).  % facing west + turn left -> facing south
turn_result(south,left,east).  % facing south + turn left -> facing east

turn_result(D0,right,D2) :- turn_result(D2,left,D0).  % reverse turn_result

% additional inheritance to simplify rules for unknown rooms
unknown(unknownN;unknownS;unknownE;unknownW).  % distinct unknowns for each cardinal direction (N,S,E,W)
instance_of(unknownN;unknownS;unknownE;unknownW, unknown).

% minigrid vocabulary and predicates (grid positions, directions, etc)
%gridpos(X,Y)
% direction vectors
dvec(north,0,-1).  % minus one to Y
dvec(south,0, 1).  %  plus one to Y
dvec(east, 1, 0).  %  plus one to X
dvec(west,-1, 0).  % minus one to X


% ------- Things (grounded instances) -------
player(player).        % always included
inventory(inventory).  % always included

goal(goal).
wall(wall).   % generic, all wall tiles are the same
lava(lava).   % generic, all lava tiles are the same

traversible(goal,0).
traversible(lava,0).

% --------------------------------------------------------------------------------
% ---- initialize step 0 values (some fluents that require inertia from step-1)

%need_to_find(find_first,0).  % initially, we're looking for the cookbook

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


% (ground truth) knownledge about inventory capacity limit
_inventory_max(1). % :- class(slot), N=#count {slot(X):slot(X)}.


% ------------------------------------------------------------------------------
% -------------------        OBSERVATION_STEP          -------------------------
% ------------------------------------------------------------------------------

% NOTE: obs_step(t) is only called after adding new facts
%    as a result of visiting a new room or opening a container for the first time
% ------------------ obs_step(t) t=[0...] ----------------------
#program obs_step(t).  % fluents that are independent of history (don't reference step-1, apply also for step 0)

%%_have_found(O,t) :- instance_of(O,thing), _visible_in_room(O,R2,t), room_changed(R,R2,t), timestep(t).
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

%pos_in_room(gpos(X,Y),R) : r(R), instance_of(gpos(X,Y),gridpos).

link(R1,D,R2) :- r(R1), r(R2), d(D), link(R1,D,unknownE), link(R2,D,unknownW).
link(R1,D,R2) :- r(R1), r(R2), d(D), link(R1,D,unknownW), link(R2,D,unknownE).
link(R1,D,R2) :- r(R1), r(R2), d(D), link(R1,D,unknownN), link(R2,D,unknownS).
link(R1,D,R2) :- r(R1), r(R2), d(D), link(R1,D,unknownS), link(R2,D,unknownN).

% ------------------------------------------------------------------------------
% -----------               EVERY_STEP_RULES           -------------------------
% ------------------------------------------------------------------------------

#program predict_step(t).

traversible(O,t) :- is_openable(O), open(O,t).
traversible(O,t) :- traversible(O,t-1), not is_openable(O).

% is the thing X close enough to the player to interact with
% in_room(O,R,t) :- at(O,P,t), instance_of(P,gridpos), pos_in_room(P,R).
% in_room(X,R,t) :- on(X,S,t), in_room(S,R,t), instance_of(S,s), r(R).
% in_room(X,R,t) :- in(X,C,t), in_room(C,R,t), instance_of(C,c), open(C,t), r(R).

%is_here(X,t) :- at(player,R,t), in_room(X,R,t), r(R).
%is_here(R,t) :- at(player,R,t), r(R).

is_here(O,t) :- pos(player,gpos(X,Y),t),
                facing(player,D0,t),
                dvec(D0,DELTA_X,DELTA_Y),
                pos(O,gpos(X+DELTA_X,Y+DELTA_Y),t).

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


% --- inertia: doors and containers don't change state unless player acts on them
open(X,t) :- is_openable(X), open(X,t-1), not act(do_close(t,X),t).
open(X,t) :- is_openable(X), act(do_open(t,X),t).
closed(X,t) :- closed(X,t-1), not act(do_open(t,X),t).
locked(X,t) :- locked(X,t-1), not act(do_unlock(t,X,_),t).
% ------ CONSTRAINTS ------
:- open(X,t), closed(X,t).  %[,is_openable(X).]    % any door or container can be either open or closed but not both
:- locked(X,t), open(X,t).  %[,is_lockable(X).]    % can't be both locked and open at the same time

%count_inventory(N,t) :- timestep(t), N=#sum{1,in(O,inventory,t):in(O,inventory,t)}.  % #sum of 1, = same as #count{}
inventory_count(t,N) :- N = #count{in(O,inventory,t):in(O,inventory,t)}.

% (unobserved) ground truth information about whether the inventory is already full
%inventory_is_full(t) :- _inventory_max(N), #count{in(O,inventory,t):in(O,inventory,t)} >= N.
_inventory_is_full(t) :- _inventory_max(M), inventory_count(t,N), N >= M.

% or based on the agent's belief state (if have observed an event that reveals inventory capacity)
inventory_is_full(t) :- inventory_capacity(M,t), inventory_count(t,N), N >= M.

did_take(O,t) :- act(do_take(t,O), t), not _inventory_is_full(t-1).
% by attempting and failing to take something, we learn inventory capacity limit
% inventory_capacity(N,t) :- act(do_take(t,X,Y), t), _inventory_is_full(t-1), inventory_count(t-1,N).
% inventory_capacity(N,t) :- inventory_capacity(N,t-1).
inventory_capacity(M,t) :- _inventory_max(M), t>=0.  % minigrid inventory capacity is known a priori

% ---- inertia: objects don't move unless moved by the player
at(X,R,t) :- at(X,R,t-1), r(R), instance_of(X,thing), not did_take(X,t).
at(O,P,t) :- at(O,P,t-1), instance_of(P,gridpos), instance_of(O,o), not did_take(O,t).
on(O,S,t) :- on(O,S,t-1), instance_of(S,s), not did_take(O,t).
in(O,C,t) :- in(O,C,t-1), instance_of(C,c), not did_take(O,t).

in(O,inventory,t) :- in(O,inventory,t-1), not act(do_put(t,O,_),t), not consumed(O,t).

in(O,inventory,t) :- did_take(O,t).  % if player takes an object, it moves to the inventory

on(O,S,t) :- act(do_put(t,O,S),t), instance_of(S,s).  % player puts an object onto a supporting object
in(O,C,t) :- act(do_put(t,O,C),t), instance_of(C,c).  % player puts an object into a container
at(O,P,t) :- act(do_put(t,O,P),t), instance_of(P,gridpos).  % player drops an object to the floor of a room


% in gridworld, opening a box destroys it (removes it from the grid)
consumed(B,t) :- act(do_open(t,B),t), box(B).  

consumed(O,t) :- consumed(O,t-1), timestep(t).


% ------------------ predict_step(t) t=[0...] ----------------------
#program predict_step(t).    % applied at each timestep >=0

% define fluents that determine whether player can move from room R0 to R1
%%%%%%%% TODO: ? add NSEW to free(R0,R1,NSEW,t)
{at(player,P,t):instance_of(P,gridpos)} = 1 :- timestep(t).   % player is at exactly one grid postion at any given time
{facing(player,NSEW,t):instance_of(NSEW,direction)} = 1 :- timestep(t).   % player is facing exactly one direction at any given time

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

% --- inertia: player stays in the same place unless acts to move to another
  % stay in the current room unless current action is do_moveP
%at(player,R0,t) :- at(player,R0,t-1), r(R0), {act(do_moveP(t,R0,R,NSEW),t):r(R),direction(NSEW)}=0. %, T<=maxT.
%at(player,R0,t) :- at(player,R0,t-1), r(R0), not act(do_moveP(t,R0,_,NSEW),t):direction(NSEW). %, T<=maxT.
at(player,R,t) :- at(player,R,t-1), r(R), not room_changed(R,_,t).

  % player moved at time t, from previous room R0 to new room R
%room_changed(R0,R,t) :- act(do_moveP(t,R0,_,NSEW),t), at(player,R0,t-1), r(R0), _connected(R0,R,NSEW), direction(NSEW). %, R!=R0.
at(player,R,t) :- room_changed(R0,R,t).

pos(player,P,t) :- pos(player,P,t-1), instance_of(P,gridpos), not pos_changed(P,_,t).
pos(player,P2,t) :- pos(player,P0,t-1), instance_of(P,gridpos), pos_changed(P0,P2,t).
pos(O,P,t) :- pos(O,P,t-1), instance_of(P,gridpos), O!=player.

facing(player,D,t) :- facing(player,D,t-1), not direction_changed(D,_,t).
facing(player,D2,t) :- facing(player,D0,t-1), direction_changed(D0,D2,t).

atP(P,t) :- pos(player,P,t), instance_of(P,gridpos).   % Alias for player's initial position
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
:- act(A,t), A != a.   % already executed actions cannot be changed

% ------------------------------------------------------------------------------
% ----------------            ACTION_STEP_RULES           ----------------------
% ------------------------------------------------------------------------------

% ------------------ step(t) t=[1...] ----------------------
#program step(t).    % applied at each timestep >=1

% Generate
timestep(t).

{act(X,t):is_action(X,t)} = 1 :- timestep(t). % player must choose exactly one action at each time step.

%{at(player,R,t):r(R)} = 1 :- timestep(t).   % player is in exactly one room at any given time


% ------------------------------------------------------------------------------
% -------------------            GAME_RULES_COMMON           -------------------
% ------------------------------------------------------------------------------


% ------ GO ------
% go/east :: at(P, r) & $west_of(r, r') & $free(r, r') & $free(r', r) -> at(P, r')
% go/north :: at(P, r) & $north_of(r', r) & $free(r, r') & $free(r', r) -> at(P, r')
% go/south :: at(P, r) & $north_of(r, r') & $free(r, r') & $free(r', r) -> at(P, r')
% go/west :: at(P, r) & $west_of(r', r) & $free(r, r') & $free(r', r) -> at(P, r')


% Test constraints
%:- at(player,R0,t-1), at(player,R,t), r(R0), r(R), R!=R0, not free(R,R0,t-1).

 % can move to a connected room, if not blocked by a closed door
% 0 {do_moveP(t,R0,R,NSEW):free(R0,R,t),connected(R0,R,NSEW),direction(NSEW)} 1 :-
%                          free(R0,R,t),connected(R0,R,NSEW),direction(NSEW), at(player,R0,t-1), r(R0). %
% 0 {do_moveP(t,R0,R,NSEW):free(R0,R,t),connected(R0,R,NSEW,t-1),direction(NSEW),at(player,R0,t-1)} 1 :-
%                          free(R0,R,t),connected(R0,R,NSEW,t-1),direction(NSEW),at(player,R0,t-1), r(R0). %

room_changed(R0,R,t) :- act(do_moveP(t,R0,_,NSEW),t), at(player,R0,t-1), r(R0), _connected(R0,R,NSEW), direction(NSEW). %, R!=R0.

0 {do_stepP(t,M):stepmove(M)} 1 . % at each turn, can (among other things) move forward or turn right or left

direction_changed(NSEW,NSEW2,t) :- act(do_stepP(t,M),t), facing(player,NSEW,t-1), stepmove(M), turn_result(NSEW,M,NSEW2).
pos_changed(gpos(X0,Y0),gpos(X0+DELTA_X, Y0+DELTA_Y),t) :- act(do_stepP(t,forward),t),
      pos(player,gpos(X0,Y0),t-1), facing(player,NSEW,t-1), dvec(NSEW,DELTA_X,DELTA_Y).


% Test constraints
  % can't step forward if there is an obstacle in the way
:- do_stepP(t,forward), pos(player,gpos(X,Y),t-1), facing(player,D0,t-1),
   dvec(D0,DELTA_X,DELTA_Y), grid_is(gpos(X+DELTA_X,Y+DELTA_Y), O), not traversible(O,t-1).

:- do_stepP(t,forward), pos(player,gpos(X,Y),t-1), facing(player,D0,t-1),
   dvec(D0,DELTA_X,DELTA_Y), pos(O,gpos(X+DELTA_X,Y+DELTA_Y), t), O!=player, not traversible(O,t-1).

is_action(do_stepP(t,M), t) :- do_stepP(t,M), stepmove(M). %, r(R1), r(R2), direction(NSEW).

% :- do_moveP(t,R0,U,NSEW),unknown(U),direction(NSEW),r(R0),timestep(t),not free(R0,U,t).  % can't go that way: not a valid action
% :- do_moveP(t,R0,R,NSEW),direction(NSEW),r(R0),r(R),timestep(t),not free(R0,R,t).  % can't go that way: not a valid action
% :- do_moveP(t,R0,U,NSEW),unknown(U),direction(NSEW),r(R0),timestep(t),room_changed(R0,R2,t),free(R0,R2,t-1).  % if dest room is known, use it explicitly
%
% is_action(do_moveP(t,R1,R2,NSEW), t) :- do_moveP(t,R1,R2,NSEW). %, r(R1), r(R2), direction(NSEW).

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

% can open a closed but unlocked door or container
% 0 {do_open(t,D):d(D),closed(D,t-1),at(player,R0,t),has_door(R0,D)} 1 :- at(player,R0,t), has_door(R0,D), d(D), closed(D,t-1), not locked(D,t-1). % R1 -- might be unknown: = not a room
0 {do_open(t,CD):is_openable(CD),closed(CD,t-1),is_here(CD,t-1)} 1 :- is_here(CD,t-1), is_openable(CD), closed(CD,t-1), not locked(CD,t-1).
% can open a closed but unlocked container
% 0 {do_open(t,C):instance_of(C,c),closed(C,t-1),is_here(C,t)} 1 :- is_here(C,t), instance_of(C,c), closed(C,t-1), not locked(C,t-1).
% Test constraints
:- do_open(t,CD), d(CD), not closed(CD,t-1). % can't open a door or container that isn't currently closed
%:- do_open(t,D), d(D), r(R), at(player,R,t), not has_door(R,D).  % can only open a door if player is in appropriate room
% have to be next to and facing a container or door to open it
:- do_open(t,C), instance_of(C,c), not is_here(C,t-1).
:- do_open(t,D), instance_of(D,d), not is_here(D,t-1).
:- do_open(t,D), instance_of(D,d), locked(D,t-1).  % can't open a locked door

is_action(do_open(t,CD), t) :- do_open(t,CD).  %, is_openable(CD).

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
%0 {do_take(t,O,C):in(O,C,t-1)} 1 :- at(player,R,t-1), r(R), instance_of(O,o), instance_of(C,c), at(C,R,t-1), in(O,C,t-1), open(C,t-1), timestep(t).
% -- take/s :: can take an object that's on a support if player is in the same room as the suppport
%0 {do_take(t,O,S):on(O,S,t-1)} 1 :- at(player,R,t-1), r(R), instance_of(O,o), instance_of(S,s), at(S,R,t-1), on(O,S,t-1), timestep(t).

% -- take :: can take an object (a portable thing) if player is in the same room and it is on the floor
0 {do_take(t,O):is_here(O,t-1),instance_of(O,o)} 1 :- is_here(O,t), instance_of(O,o), timestep(t).


% don't try to pick up anything if already carrying inventory_max items (if inventory_max is known)
:- do_take(t,_), timestep(t), inventory_is_full(t-1).

is_action(do_take(t,O),t) :- do_take(t,O).


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


% ------------------------------------------------------------------------------
% ------------------               COOKING_RULES           ---------------------
% ------------------------------------------------------------------------------

#program cooking_step(t).


%---------RECIPE_NEED_TO_FIND-----------
need_to_acquire(F,t) :- in_recipe(F), not in(F,inventory,t), not consumed(F,t), not have_acquired(F,t). % if t-1, inertia rule applies
:- need_to_acquire(F,t), consumed(F,t).

% need_to_acquire(O,t) :- in_recipe(I,F), should_cut(I,V), cuttable(F),
%      sharp(O), not cut_state(F,V,t), not have_acquired(O,t), timestep(t).

% need_to_find(A,t) :-
%     in_recipe(I,F), should_cook(I,grilled), cookable(F), not cooked_state(F,grilled,t-1), toaster(A), not have_found(A,t), timestep(t).
% need_to_find(A,t) :-
%      in_recipe(I,F), should_cook(I,roasted), cookable(F), not cooked_state(F,roasted,t-1), oven(A), not have_found(A,t), timestep(t).
% need_to_find(A,t) :-
%      in_recipe(I,F), should_cook(I,fried), cookable(F), not cooked_state(F,fried,t-1), stove(A), not have_found(A,t), timestep(t).

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

new_room(t) :- {first_visited(R,t):r(R)}>0.
newly_opened(t) :- {first_opened(C,t):instance_of(C,c)}>0.
new_acquire(t) :- {first_acquired(O,t):need_to_acquire(O,t-1)}>0.
inventory_max_determined(t) :- inventory_capacity(M,t), not inventory_capacity(M,t-1), query(t).

% % block cheating based on GT knowledge of _inventory_max(M)
% :- query(t), inventory_max_determined(t),
%       act(do_take(t,X,_),t), not need_to_acquire(X,t).

goal1_achieved(t) :- pos(player,P,t), grid_is(P,goal), query(t). %recipe_read(t), not recipe_read(t-1).
goal1_has_been_achieved(t) :- goal1_achieved(t).
goal1_has_been_achieved(t) :- goal1_has_been_achieved(t-1).

goal2_achieved(t) :- goal1_has_been_achieved(t). %consumed(meal_0,t).

solved_all(t) :- goal2_achieved(t), query(t).

search_step(t) :- new_room(t), query(t).
search_step(t) :- newly_opened(t), query(t).

acquire_step(t) :- new_acquire(t), query(t).
acquire_step(t) :- inventory_max_determined(t), query(t).

% ------------------------------------------------------------------------------
% Constraints (implement prioritization of subgoals relative to each other)

% if we don't need to gather stuff, inventory capacity is irrelevant
:- not need_to_gather(t), inventory_max_determined(t), query(t).

:- can_acquire(t), search_step(t), queryt(t).
% prefer greedy/1-step acquire over exploring a aew room
%:- can_acquire(t), new_room(t), query(t).
% prefer greedy/1-step acquire over opening a new container
%:- can_acquire(t), newly_opened(t), query(t).


% prefer opening a container over exploring a aew room
:- need_to_search(t), can_open(t), new_room(t), query(t).

% enforce strict sequential ordering: need to read the recipe before preparing the meal
:- goal2_achieved(t), not goal1_has_been_achieved(t), query(t).

% if we know where the recipe is, FAIL UNTIL WE READ the Recipe
:- not goal1_has_been_achieved(t), not need_to_search(t), t>0, query(t).

% if searching or gathering is not required, fail until we achieve our main goal
:- goal1_has_been_achieved(t-1), not goal2_achieved(t),
     not need_to_search(t), not need_to_gather(t),
 query(t).

%% UNTIL WE FIND THE RECIPE: allow solver success for search (exploring unseen locations)
:- not goal1_has_been_achieved(t),
     need_to_search(t),
         not search_step(t), % not new_room(t), not newly_opened(t),
         not inventory_max_determined(t),
 query(t).

% ****** After finding the recipe: ******

% Fail unless/until we gather a required item
:- goal1_has_been_achieved(t-1), not goal2_achieved(t),
     not need_to_search(t), need_to_gather(t),
          not acquire_step(t), % not new_acquire(t), not inventory_max_determined(t),
 query(t).

% Fail unless/until we explore a new room or container
:- goal1_has_been_achieved(t-1), not goal2_achieved(t),
     need_to_search(t), not need_to_gather(t),
           not search_step(t), % not new_room(t), not newly_opened(t),
 query(t).
 
% Allow success upon exploring or gathering, if appropriate
:- goal1_has_been_achieved(t-1), not goal2_achieved(t),
     need_to_search(t), need_to_gather(t),
         not search_step(t),  % not new_room(t), not newly_opened(t),
         not acquire_step(t), % not new_acquire(t), not inventory_max_determined(t),
 query(t).
% NOTE (GVS ?unknown why the following don't work vs. new_xxx(t) which do) ----
%      not first_visited(_,t), not first_opened(_,t), not first_acquired(_,t), query(t).


%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#show atP/2.
#show act/2.
#show facing/3.
%#show traversible/2.

%#show did_act/2.
%#show atP/2.
%#show A: act(A,T),timestep(T).
%#show inventory_capacity/2.
%#show inventory_max_determined/1.
%#show inventory_is_full/1.
%#show _inventory_is_full/1.
%#show can_take/2.

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
