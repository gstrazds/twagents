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
#const step_max_mins=3.  % (default value)
#const step_max_secs=30. % (default value)  Give up if a solving step takes > step_max_mins:step_max_secs

#script (python)
import datetime
from clingo import Function, Symbol, String, Number, SymbolType

def get(val, default):
    return val if val != None else default

def main(prg):

    imin = get(prg.get_const("imin"), Number(1))
    imax = get(prg.get_const("imax"), Number(500))
    istop = get(prg.get_const("istop"), String("SAT"))

    MIN_PRINT_STEP = 0  # print elapsed time for each solving step >= this value
    STEP_MAX_MINS = get(prg.get_const("step_max_mins"), Number(2)).number
    STEP_MAX_SECS = get(prg.get_const("step_max_secs"), Number(30)).number
    STEP_MAX_ELAPSED_TIME = datetime.timedelta(minutes=STEP_MAX_MINS, seconds=STEP_MAX_SECS)

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
                print(f"  {'++' if step == t else '--'} [{t}] : {str_atom}")
                _actions_list.append(atom)
            elif atom.name == 'recipe_read' \\
              and len(atom.arguments)==1 and atom.arguments[0].type is SymbolType.Number:
                if atom.arguments[0].number == step:
                    print(f"  ++ {atom}")
                _newly_discovered_facts.append("cookbook")
            elif atom.name == 'solved_all' \\
              and len(atom.arguments)==1 and atom.arguments[0].type is SymbolType.Number:
                print(f"  ++! {atom}")
                _newly_discovered_facts.append("solved_all")
                _solved_all = True
            elif (atom.name == 'first_visited' \\
               or atom.name == 'first_opened'  \\
               or atom.name == 'first_acquired') \\
              and len(atom.arguments)==2 and atom.arguments[1].type is SymbolType.Number:
                if atom.arguments[1].number == step:    #, f"[{step}] {atom.name} {atom.arguments[1]}"
                    print(f"  ++ {atom}")
                    _newly_discovered_facts.append(str(atom.arguments[0]))
                else:
                    print(f"  -- {atom}")
        if _solved_all:
            return False
        return True  # False -> stop after first model

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
        if ret and ret.satisfiable:
            for _name in _newly_discovered_facts:
                if _name == "solved_all":
                    solved_all = True
                elif _name == "cookbook":
                    if not _recipe_read:
                        _recipe_newly_seen = True
                        print(f"+++++ _recipe_newly_seen: ADDING #program recipe({step-1})")
                        parts.append(("recipe", [Number(step-1)]))
                        parts.append(("cooking_step", [Number(step-1)]))  # this once will have cooking_step(both step and step-1)
                    _recipe_read = True
                elif _name.startswith("r_"):    # have entered a previously unseen room
                    print(f"ADDING #program room_{_name} ({step-1}).")
                    parts.append((f"room_{_name}", [Number(step-1)]))
                    parts.append(("obs_step", [Number(step-1)]))
                elif _name.startswith("c_"):    # opened a container for the first time
                    print(f"OBSERVING CONTENTS OF {_name} ({step-1}).")
                    #parts.append((f"c_{_name}", [Number(step-1)]))
                    parts.append(("obs_step", [Number(step-1)]))
                else:
                    print("%%%%% IGNORING FIRST ACQUIRED:", _name)
            if solved_all:
                break      # stop solving immediately
        if step == 0:
            parts.append(("base", []))
            #parts.append(("recipe", [Number(step)]))
            parts.append(("initial_state", [Number(0)]))
            parts.append(("initial_room", [Number(0)]))     #(f"room_{first_room}", [Number(0)]))
            #parts.append(("obs_step", [Number(0)]))  #step==0
            parts.append(("predict_step", [Number(0)]))
            parts.append(("check", [Number(step)]))     #step==0
        else:  #if step > 0:

            if len(_actions_facts):
                actions_facts_str = "\\n".join(_actions_facts)
                actions_facts_str = actions_facts_str.replace("act(", "did_act( ")
                print(f"\\n+++++ ADDING prev_actions: +++\\n{actions_facts_str}\\n----", flush=True)
                #print("\\t", "\\n\\t".join(_actions_facts), flush=True)
                prg.add("prev_actions", [], actions_facts_str)
                parts.append(("prev_actions", []))

            #parts.append(("obs_step", [Number(step)]))
            parts.append(("predict_step", [Number(step)]))
            parts.append(("step", [Number(step)]))
            if _recipe_read:
                print(f"+ ADDING #program cooking_step({step})")
                parts.append(("cooking_step", [Number(step)]))
            parts.append(("check", [Number(step)]))

            #  query(t-1) becomes permanently = False (removed from set of Externals)
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
        if elapsed_time > STEP_MAX_ELAPSED_TIME:
            print(f"--- [{step}] Step time {elapsed_time} > {STEP_MAX_ELAPSED_TIME} ... Stop solving.")
            break
        step = step+1

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
        '% ------- Types -------',
        '',
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
         '% ------- Facts -------',
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


def generate_ASP_for_game(game, asp_file_path, hfacts=None, no_python=False):
    if not hfacts:
        _inform7 = Inform7Game(game)
        hfacts = list(map(_inform7.get_human_readable_fact, game.world.facts))

    game_definition = convert_to_asp(game, hfacts)

    with open(asp_file_path, "w") as aspfile:
        aspfile.write("#const find_first=o_0.   % the cookbook is always o_0")
        aspfile.write('\n')
        if not no_python:
            aspfile.write(INCREMENTAL_SOLVING) # embedded python loop for solving TW games

        aspfile.write(game_definition)   # initial state of one specific game
        # ---- GAME DYNAMICS               # logic/rules common to all games
        source_path = Path(__file__).resolve()
        source_dir = source_path.parent
        # print("SOURCE DIRECTORY:", source_dir)
        with open(source_dir / 'tw_asp.lp', "r") as rulesfile:
            aspfile.write(rulesfile.read())
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

        # ##aspfile.write(":- movedP(T,R,R1), at(player,R1,T0), timestep(T0), T0<T .  % disallow loops\n")
        # ## For branch & bound optimization:
        # ## aspfile.write( #":- not at(player,r_0,maxT).  % end up in the kitchen\n")
        # ##     "ngoal(T) :- at(player,R,T), r(R), R!=r_0 .  % want to end up in the kitchen (r_0)\n" \
        # ##     ":- ngoal(maxT).\n  % anti-goal -- fail if goal not achieved"
        # ## )
        # ##aspfile.write("_minimize(1,T) :- ngoal(T).\n")
