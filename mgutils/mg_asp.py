import re
from pathlib import Path
from typing import Optional, Any, Tuple, List
from operator import itemgetter

from minigrid.minigrid_env import MiniGridEnv

from minigrid.core.constants import IDX_TO_COLOR, COLOR_TO_IDX
from minigrid.core.constants import IDX_TO_OBJECT, OBJECT_TO_IDX, STATE_TO_IDX, DIR_TO_VEC
from minigrid.core import Grid
from minigrid.core.world_object import WorldObj

INCREMENTAL_SOLVING = \
"""% #include <incmode>.
#const imax=500.  % (default value for) give up after this many iterations
#const step_max_mins=3.  % (default value)
#const step_max_secs=30. % (default value)  Give up if a solving step takes > step_max_mins:step_max_secs

#script (python)
# from mgutils.mg_asp_incremental import mg_solve_incremental
# import datetime
# from clingo import Function, Symbol, String, Number, SymbolType

def get(val, default):
    return val if val != None else default

def main(prg):

    imin = get(prg.get_const("imin"), Number(1))
    imax = get(prg.get_const("imax"), Number(500))
    
    MIN_PRINT_STEP = 0  # print elapsed time for each solving step >= this value
    STEP_MAX_MINS = get(prg.get_const("step_max_mins"), Number(2)).number
    STEP_MAX_SECS = get(prg.get_const("step_max_secs"), Number(30)).number
    STEP_MAX_ELAPSED_TIME = timedelta(minutes=STEP_MAX_MINS, seconds=STEP_MAX_SECS)

    mg_solve_incremental(prg, 
            istop="SAT", imin=imin.number, imax=imax.number,
            step_max_time=STEP_MAX_ELAPSED_TIME, min_print_step=MIN_PRINT_STEP)
#end.
"""

# NOTE: this is how to get the object currently in front of the agent:
def get_focus_obj(minigrid_env) -> Optional[WorldObj]:
    # Get the position in front of the agent
    fwd_pos = minigrid_env.front_pos

    # Get the contents of the cell in front of the agent
    fwd_obj = minigrid_env.get(*fwd_pos)
    return fwd_obj



# OBJECT_TO_IDX = {
#     "unseen": 0,
#     "empty": 1,
#     "wall": 2,
#     "floor": 3,
#     "door": 4,
#     "key": 5,
#     "ball": 6,
#     "box": 7,
#     "goal": 8,
#     "lava": 9,
#     "agent": 10,
# }

    # if obj_type == "wall":
    #     v = Wall(color)
    # elif obj_type == "floor":
    #     v = Floor(color)
    # elif obj_type == "ball":
    #     v = Ball(color)
    # elif obj_type == "key":
    #     v = Key(color)
    # elif obj_type == "box":
    #     v = Box(color)
    # elif obj_type == "door":
    #     v = Door(color, is_open, is_locked)
    # elif obj_type == "goal":
    #     v = Goal()
    # elif obj_type == "lava":
    #     v = Lava()
    # else:
    #     assert False, "unknown object type in decode '%s'" % obj_type


# Define a WorldObj subclass to represent the player
#  (not used by MiniGrid code, only by our wrapper)
class Agent(WorldObj):
    """
    Represents the player
    """

    def __init__(self, color: str = "red"):
            super().__init__(type="agent", color=color)
            # Initial position of the object
 
    def can_contain(self) -> bool:
        """Can this contain another object?"""
        return True


# The following strings are used with string.format(),
#  given either an incrementing index if 'N:d' or X,Y from a grid posisition (X,Y)
# The gridpos version is used for objects that are static (can never be moved)

OBJ_ID_FORMAT = {
    "wall": 'w_{X:d}_{Y:d}',
    "floor": 'fl_{X:d}_{Y:d}',  # not actually used in most minigrid envs
    "door": 'd_{N:d}',
    "key": 'k_{N:d}',
    "ball": 'o_{N:d}',  # alignment w/TextWorld: balls are just generic objects [b_{N:d}]
    "box": 'c_{N:d}',
    "goal": 'g_{N:d}',
    "lava": 'l_{X:d}_{Y:d}',
    "agent": 'a_{N:d}',
}

STATIC_OBJ_TYPES = {
    "wall",
    "floor",  # not actually used in most minigrid envs
    "door",   # doors can be unlocked/opened/closed, but cannot be moved
    "goal",  # ? can goals ever move around?
    "lava",
}
def is_static_obj(obj:WorldObj):
    return obj.type in STATIC_OBJ_TYPES


STATIC_FACTS = ['cuttable', 'cookable', 'sharp', 'cooking_location',
                'link', 'north_of', 'east_of', 'south_of', 'west_of']

OPEN_STATE = ['open', 'closed', 'locked'] #anything other than open -> -open
LOCATION_REL = ['at', 'on', 'in']
CONNECTION_REL = ['east_of', 'west_of', 'north_of', 'south_of']
LINK_REL = ['link', 'free']


def is_fluent_fact(fact_name: str, args_str: Optional[str] = None):
    if fact_name in STATIC_FACTS:
        return False
    return True


_OBJ_INFOS = '_obj_infos'
def get_obj_infos(obj_index):
    obj_infos = obj_index.get(_OBJ_INFOS, None)
    assert obj_infos is not None
    return obj_infos

def _assign_id_to_worldobj(obj, obj_index,
         x:int=-1,
         y:int=-1,
         id_str:str=None,
         has_been_seen=None):
    if id_str:
        obj_id = id_str
    else:
        assert not hasattr(obj, "id"), f"{str(obj)} already has id={obj.id}"
        idx = len(obj_index[obj.type])
        assert obj not in obj_index[obj.type], str(obj)
        obj_index[obj.type].append(obj)
        id_format = OBJ_ID_FORMAT.get(obj.type, None)
        if id_format:
            if "N:d" in id_format:
                id_str = id_format.format(N=idx)
            else:
                if x < 0 and y< 0:
                    assert hasattr(obj, "init_pos"), f"{obj}"
                    assert obj.init_pos is not None, f"{obj} {obj.cur_pos} {obj.init_pos}"
                    x,y = obj.init_pos
                id_str = id_format.format(X=x, Y=y)
        else:
            id_str = f"{obj.type}_{idx}"
        obj.id = id_str
    obj_infos = get_obj_infos(obj_index)
    assert id_str not in obj_infos, id_str
    obj_infos[id_str] = obj
    if has_been_seen is not None:
        obj.has_been_seen = has_been_seen
    return id_str

def assign_obj_ids_for_ASP(env:MiniGridEnv, reset_all_unseen=True):
    obj_index = {
        key:[] for key in OBJ_ID_FORMAT.keys()
    }
    obj_index[_OBJ_INFOS] = {}  # map of obj.id -> info about the object (an instance of WorldObj)
    agent_obj = agent_obj = Agent()
    agent_obj.has_been_seen = True
    agent_obj.init_pos = env.agent_pos
    agent_obj.cur_pos = env.agent_pos
    _assign_id_to_worldobj(agent_obj, obj_index, id_str='player', has_been_seen=True)

    if env.carrying:
        carried_obj = env.carrying
        agent_obj.contains = carried_obj
        obj_id = _assign_id_to_worldobj(carried_obj, obj_index, has_been_seen=True)

    if reset_all_unseen:
        has_been_seen = False
    elif reset_all_unseen is None:
        has_been_seen = None
    else:
        has_been_seen = True

    for y in range(env.grid.height):
        for x in range(env.grid.width):
            o = env.grid.get(x,y)
            if o is not None:
                _assign_id_to_worldobj(o, obj_index, x=x, y=y, has_been_seen=has_been_seen)

    return obj_index


def update_player_obj(obj_index, env:MiniGridEnv):
    # because minigrdd does not treat the aggent like a WorldDbj
    # we implement this more OO approach here
    # ASSUMES: have previously initialized a 'player' object in assign_obj_ids_for_ASP()
    agent_pos = env.agent_pos
    player_obj = get_obj_infos(obj_index).get('player', None)
    assert player_obj is not None, "EXPECTING _infos['player'] initialized in assign_obj_ids_for_ASP()"
    player_obj.cur_pos = agent_pos
    if env.carrying:
        player_obj.contains = env.carrying
    else:
        player_obj.contains = None
    return agent_pos


def get_obj_id(obj, obj_index):
    if hasattr(obj, "id"):
        return obj.id
    else:
        assert False, f"OBJ {obj} has no id!"


def get_facts_from_minigrid(env:MiniGridEnv, obj_index, timestep:int=0):
    w = env.grid.width
    h = env.grid.height
    static_facts_list = []
    fluent_facts_list = []
    agent_pos = update_player_obj(obj_index, env)
    timestep = env.step_count

    fluent_facts_list.append( f"at(player,{agent_pos[0]}, {agent_pos[1]}, {timestep})." )
    if env.carrying:
        obj = env.carrying
        fluent_facts_list.append( f"in({get_obj_id(obj, obj_index)},inventory,{timestep})." )

    for x in range(w):
        for y in range(h):
            gpos_fact = f"gpos({x},{y})."
            static_facts_list.append( gpos_fact )
            obj = env.grid.get(x,y)
            if obj is not None:
                if is_static_obj(obj):
                    fact_str = f"grid_is(gpos({x},{y}),{obj.type})."
                    static_facts_list.append(fact_str)
                    # fact_str = f"grid_obj(gpos({x},{y}),{get_obj_id(obj, obj_index)})."
                    # static_facts_list.append(fact_str)
                else:
                    fact_str = f"at({get_obj_id(obj, obj_index)},gpos({x},{y}),{timestep})."
                    fluent_facts_list.append( fact_str )
    # print("static_facts:", static_facts_list)
    # print("fluent_facts", fluent_facts_list)
    static_facts_dict = {fact_str:fact_str for fact_str in static_facts_list}
    fluent_facts_dict = {fact_str:fact_str for fact_str in fluent_facts_list}
    # for TextWorld, analogous method produces str:Proposition mappings
    return static_facts_dict, fluent_facts_dict



def convert_minigrid_to_asp(env, obj_index):

    static_facts, initial_fluents = get_facts_from_minigrid(env, obj_index, timestep=0)
 
    room_facts = {}
    # directions_map = {}
    # for fact in game.world.facts:
    #     if fact.name in CONNECTION_REL: # east_of(R1,R2), north_of, etc...
    #         r1 = fact.arguments[0].name
    #         r2 = fact.arguments[1].name
    #         assert _is_a_room(r1), str(fact)
    #         assert _is_a_room(r2), str(fact)
    #         direction = fact.name[0].upper()
    #         if r2 in directions_map:
    #             directions_map[r2][r1] = direction
    #         else:
    #             directions_map[r2] = {r1: direction}
    # print(directions_map)
    # for fact, hfact in zip(game.world.facts, hfacts):
    #     fact_str, is_part_of_recipe, is_static_fact = fact_to_asp(fact, hfact, step='t')
    #     # if is_part_of_recipe:
    #     #     recipe_facts[fact_str] = fact
    #     if is_static_fact and not fact.name in CONNECTION_REL and not fact.name in LINK_REL:
    #         static_facts[fact_str] = fact
    #     else:
    #         if fact.name in CONNECTION_REL:
    #             _transition = "_"+fact_str
    #             static_facts[_transition] = fact  # privileged knowledge for transitions in partially observable world
    #             regex_direction_of = re.compile(r'\(r_(\d)+')      # dir_of(r_1, r_2) => dir_of(unkown, r_2)
    #             unknownStr = '(unknown'+fact.name[0].upper()
    #             fact_str = regex_direction_of.sub(unknownStr, fact_str)
    #             print(fact_str)
    #         elif fact.name == 'link':
    #             regex_link = re.compile(r', r_(\d)+\)')             # link(r_1, d_N, r_2)
    #             r1 = fact.arguments[0].name
    #             r2 = fact.arguments[2].name
    #             unknownStr = f", unknown{directions_map[r1][r2]})"
    #             fact_str = regex_link.sub(unknownStr, fact_str)  #  => link(r_1, d_N, unknown)
    #             print(fact_str)
    #         elif fact.name == 'free':
    #             #regex_free = re.compile(r', r_(\d)+, ') #.sub(', unknown, ')
    #             regex_free = re.compile(r'^free')             # free(r_1, r_2, t)
    #             fact_str = regex_free.sub('%free', fact_str)  #  => %free(r_1, r_2, t)
    #             print(fact_str)
    #         initial_fluents[fact_str] = fact
    _initial_room = 'r_0'  #group_facts_by_room(initial_fluents, room_facts)


    asp_lines = [
        # '% ======= Types ======',
        # '% ------- IS_A -------',
    ]

    asp_lines += [
        '', # single empty line
        '% ------- Things -------'
    ]

    # for info in game._infos.values():
    #     asp_lines.append(info_to_asp(info))

    asp_lines += [
        '',  # emtpy line
         '% ======= Facts =======',
        '\n'.join(static_facts.keys()),
        '\n',
        "% ------- initial fluents (initialized with t=0) -------",
        "#program initial_state(t).",
        '',
        #'\n'.join(initial_fluents.keys()),
        '\n'.join(initial_fluents.keys()),
        '\n',
        "% ------- ROOM fluents -------",
        '',
        f"#program initial_room(t).",
        # '\n'.join([r_fact for r_fact in room_facts[_initial_room] if r_fact.startswith("at(player,")]),
        # f"room_changed(nowhere,{_initial_room},t).",
        '',
    ]
    for room in sorted(room_facts.keys()):
        asp_lines.append(f"#program room_{room}(t).")
        for r_fact in room_facts[room]:
            if not r_fact.startswith("at(player,"):
                asp_lines.append(r_fact)
        asp_lines.append('')  #an empty line after each set of room facts

    asp_str = '\n'.join(asp_lines) + '\n'
    return asp_str


def generate_ASP_for_minigrid(env, obj_index,
        seed=None,
        standalone=False,
        emb_python=None,
        ) -> str:

    env.reset(seed=seed) # ensure initial state (if seed is None, will be different each time)

    game_definition = convert_minigrid_to_asp(env, obj_index)

    if standalone and emb_python is None:
        emb_python = True
    # ---- GAME DYNAMICS               # logic/rules common to all games
    source_path = Path(__file__).resolve()
    source_dir = source_path.parent
    # print("SOURCE DIRECTORY:", source_dir)
    if standalone:
        with open(source_dir / 'mg_asp.lp', "r") as rulesfile:
            asp_rules = rulesfile.read()
            game_definition += '\n'
            game_definition += asp_rules

    if emb_python:
        with open(source_dir / 'mg_asp_incremental.py', "r") as embedpython:
            game_definition += '\n#script (python)\n'
            game_definition +=  embedpython.read()
            game_definition += '\n#end.\n'
            game_definition += INCREMENTAL_SOLVING # embedded python loop for solving TW games

    return game_definition

# def lookup_human_readable_name(asp_id:str, infos) -> str:
#     def _get_name(info):
#         return info.name if info.name else info.id
#     info = infos.get(asp_id, None)
#     if info:
#         return _get_name(info)
#     assert False, f"FAILED TO FIND info for {asp_id}"
#     return asp_id   # Fallback if FAIL

# def tw_command_from_asp_action(action, infos) -> str:
#     """
#       -- special cases
#       do_look(t,R)            -- look
#       do_moveP(t,R1,R2,NSEW)  -- go NSEW
      
#       do_take(t,O,CR)         -- take O [from C (unless CR is a room)]
#       do_put(t,O,C)           -- put O into C
#       do_put(t,O,S)           -- put O on S
#       do_put(t,O,R)           -- drop O

#       -- regular pattern: verb obj1 [obj2]
#       do_examine(t,O)
#       do_eat(t,F)
#       do_drink(t,F)
#       do_open(t,CD)
#       do_cook(t,X,A)    - cook X with A

#       do_cut(t,V,F,O)   - V F with O
#     """

#     str_action = str(action)   # for logging and debugging
#     if not action.name.startswith("do_"):
#         assert False, f"Unexpected action: {str_action}"
#     if action.name == 'do_make_meal':
#         return "prepare meal"
#     # by default, simply remove the "do_" prefix
#     verb = action.name[3:]
#     if verb == "moveP":  # special case
#         verb = "go"
#         assert len(action.arguments) == 4, str_action
#         direction = action.arguments[3].name
#         return f"go {direction}"
#     num_args = len(action.arguments)
#     assert num_args >= 1, str_action
#     if num_args == 1:
#         return verb

#     obj2_name = ''
    
#     if num_args > 4:
#         assert False, "UNEXPECTED: "+str_action
#     if num_args == 4:
#         assert verb == 'cut', str_action
#         verb = action.arguments[1]
#         id1 = action.arguments[2].name
#         id2 = action.arguments[3].name
#     else:  #num_args >= 2:    # 1st arg is timestep, 2nd is an entity id
#         id1 = action.arguments[1].name
#         if num_args > 2:
#             id2 = action.arguments[2].name
#         else:
#             id2 = None
#     obj1_name = lookup_human_readable_name(id1, infos)
#     if id2:
#         obj2_name = lookup_human_readable_name(id2, infos)
#     else:
#         obj2_name = ''

#     preposition = ''
#     if verb == 'put':
#         if id2.startswith('r_'):
#             verb = 'drop'
#             obj2_name = ''   # don't mention the room (target is the floor) 
#         elif id2.startswith('c_') or id2.startswith('oven_'):
#             preposition = ' into '
#         elif id2.startswith('s_') or id2.startswith('stove_'):
#             preposition = ' on '
#         else:
#             assert False, "Unexpected target for "+str_action
#     elif verb == 'take':
#         if id2.startswith('r_'):
#             obj2_name = ''   # don't mention the room (object is on the floor)
#         else:
#             preposition = ' from '
#     elif obj2_name:
#         preposition = ' with '
#     return f"{verb} {obj1_name}{preposition}{obj2_name}"

