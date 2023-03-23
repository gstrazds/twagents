import re
from pathlib import Path
from typing import Optional, Any, Tuple
from operator import itemgetter

from textworld.logic import Proposition, Variable
from textworld.generator import Game, KnowledgeBase
from textworld.generator.vtypes import VariableType, VariableTypeTree
from textworld.generator.inform7 import Inform7Game


INCREMENTAL_SOLVING = \
"""% #include <incmode>.
#const imax=500.  % (default value for) give up after this many iterations
#const step_max_mins=10.  % (default value)
#const step_max_secs=1. % (default value)  Give up if a solving step takes > step_max_mins:step_max_secs

#script (python)
#from twutils.tw_asp_incremental import tw_solve_incremental
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

    tw_solve_incremental(prg, 
            istop="SAT", imin=imin.number, imax=imax.number,
            step_max_time=STEP_MAX_ELAPSED_TIME, min_print_step=MIN_PRINT_STEP)

#end.
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

def convert_to_asp(game, hfacts):
 
    type_infos = types_to_asp(game.kb.types)
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


    asp_lines = [
        "#const find_first=o_0.   % the cookbook is always o_0",
        '',

        '% ======= Types ======',
        '% ------- IS_A -------',
    ]

    for typename, _ in type_infos:
        # asp_str.write(f"class({typename}). ") . # can derive this automatically from instance_of() or subclass_of()
        asp_lines.append(f"instance_of(X,{typename}) :- {typename}(X).")
    for typename, parent_type in type_infos:
        if parent_type:  
            # and parent_type != 'thing':  # currently have no practical use for 'thing' base class (dsintinugishes objects from rooms)
            asp_lines.append(f"subclass_of({typename},{parent_type}).")

    asp_lines += [
        '', # single empty line
        '% ------- Things -------'
    ]

    for info in game._infos.values():
        asp_lines.append(info_to_asp(info))

    asp_lines += [
        '',  # emtpy line
         '% ======= Facts =======',
        '\n'.join(static_facts.keys()),
        '\n',
        "% ------- initial fluents (initialized with t=0) -------",
        "#program initial_state(t).",
        '',
        '\n'.join(initial_fluents.keys()),
        '\n',
        "% ------- Recipe -------",
        "#program recipe(t).",
        '',
        '\n'.join(recipe_facts.keys()),
        '',
        "in_recipe(I,F) :- ingredient(I), in(I,recipe), base(F,I), instance_of(F,f).",
        "in_recipe(F) :- in_recipe(I,F).",
        '\n',
        "% ------- ROOM fluents -------",
        '',
        f"#program initial_room(t).",
        '\n'.join([r_fact for r_fact in room_facts[_initial_room] if r_fact.startswith("at(player,")]),
        f"room_changed(nowhere,{_initial_room},t).",
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


def generate_ASP_for_game(game, asp_file_path, hfacts=None, standalone=False, emb_python=False):
    if not hfacts:
        _inform7 = Inform7Game(game)
        hfacts = list(map(_inform7.get_human_readable_fact, game.world.facts))

    game_definition = convert_to_asp(game, hfacts)

    # ---- GAME DYNAMICS               # logic/rules common to all games
    source_path = Path(__file__).resolve()
    source_dir = source_path.parent
    # print("SOURCE DIRECTORY:", source_dir)
    if standalone:
        with open(source_dir / 'tw_asp.lp', "r") as rulesfile:
            asp_rules = rulesfile.read()
            game_definition += '\n'
            game_definition += asp_rules

        #aspfile.write(TYPE_RULES)
        #aspfile.write(OBSERVATION_STEP)
        #aspfile.write(EVERY_STEP_MAP_RULES)
        #aspfile.write(EVERY_STEP_RULES)

        #aspfile.write(ACTION_STEP_RULES)
        #aspfile.write(GAME_RULES_COMMON)
        #aspfile.write(GAME_RULES_NEW)
        #aspfile.write(COOKING_RULES)
        #aspfile.write(RECIPE_NEED_TO_FIND)

        #aspfile.write(CHECK_GOAL_ACHIEVED)

    else:  #if not standalone:
        asp_file_path = str(asp_file_path)
        if asp_file_path and asp_file_path.endswith(".lp") \
            and not asp_file_path.endswith(".0.lp"):
            asp_file_path = asp_file_path[0:-3] + ".0.lp"
    if emb_python:
        with open(source_dir / 'tw_asp_incremental.py', "r") as embedpython:
            game_definition += '\n#script (python)\n'
            game_definition +=  embedpython.read()
            game_definition += '\n#end.\n'
            game_definition += INCREMENTAL_SOLVING # embedded python loop for solving TW games

    if asp_file_path:
        with open(asp_file_path, "w") as aspfile:
            aspfile.write(game_definition)   # initial state of one specific game
    return game_definition

def lookup_human_readable_name(asp_id:str, infos) -> str:
    def _get_name(info):
        return info.name if info.name else info.id
    info = infos.get(asp_id, None)
    if info:
        return _get_name(info)
    assert False, f"FAILED TO FIND info for {asp_id}"
    return asp_id   # Fallback if FAIL

def tw_command_from_asp_action(action, infos) -> str:
    """
      -- special cases
      do_look(t,R)            -- look
      do_moveP(t,R1,R2,NSEW)  -- go NSEW
      
      do_take(t,O,CR)         -- take O [from C (unless CR is a room)]
      do_put(t,O,C)           -- put O into C
      do_put(t,O,S)           -- put O on S
      do_put(t,O,R)           -- drop O

      -- regular pattern: verb obj1 [obj2]
      do_examine(t,O)
      do_eat(t,F)
      do_drink(t,F)
      do_open(t,CD)
      do_cook(t,X,A)    - cook X with A

      do_cut(t,V,F,O)   - V F with O
    """

    str_action = str(action)   # for logging and debugging
    if not action.name.startswith("do_"):
        assert False, f"Unexpected action: {str_action}"
    if action.name == 'do_make_meal':
        return "prepare meal"
    # by default, simply remove the "do_" prefix
    verb = action.name[3:]
    if verb == "moveP":  # special case
        verb = "go"
        assert len(action.arguments) == 4, str_action
        direction = action.arguments[3].name
        return f"go {direction}"
    num_args = len(action.arguments)
    assert num_args >= 1, str_action
    if num_args == 1:
        return verb

    obj2_name = ''
    
    if num_args > 4:
        assert False, "UNEXPECTED: "+str_action
    if num_args == 4:
        assert verb == 'cut', str_action
        verb = action.arguments[1]
        id1 = action.arguments[2].name
        id2 = action.arguments[3].name
    else:  #num_args >= 2:    # 1st arg is timestep, 2nd is an entity id
        id1 = action.arguments[1].name
        if num_args > 2:
            id2 = action.arguments[2].name
        else:
            id2 = None
    obj1_name = lookup_human_readable_name(id1, infos)
    if id2:
        obj2_name = lookup_human_readable_name(id2, infos)
    else:
        obj2_name = ''

    preposition = ''
    if verb == 'put':
        if id2.startswith('r_'):
            verb = 'drop'
            obj2_name = ''   # don't mention the room (target is the floor) 
        elif id2.startswith('c_') or id2.startswith('oven_'):
            preposition = ' into '
        elif id2.startswith('s_') or id2.startswith('stove_'):
            preposition = ' on '
        else:
            assert False, "Unexpected target for "+str_action
    elif verb == 'take':
        if id2.startswith('r_'):
            obj2_name = ''   # don't mention the room (object is on the floor)
        else:
            preposition = ' from '
    elif obj2_name:
        preposition = ' with '
    return f"{verb} {obj1_name}{preposition}{obj2_name}"



# ##aspfile.write(":- movedP(T,R,R1), at(player,R1,T0), timestep(T0), T0<T .  % disallow loops\n")
# ## For branch & bound optimization:
# ## aspfile.write( #":- not at(player,r_0,maxT).  % end up in the kitchen\n")
# ##     "ngoal(T) :- at(player,R,T), r(R), R!=r_0 .  % want to end up in the kitchen (r_0)\n" \
# ##     ":- ngoal(maxT).\n  % anti-goal -- fail if goal not achieved"
# ## )
# ##aspfile.write("_minimize(1,T) :- ngoal(T).\n")
