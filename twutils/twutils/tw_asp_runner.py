from typing import List, Sequence, Tuple, Optional
from operator import itemgetter
from datetime import datetime, timedelta

from clingo.control import Control
from clingo import Function, String, Number, Symbol, SymbolType

Parts = Sequence[Tuple[str, List[Symbol]]]
Externals = Sequence[Tuple[Symbol, Optional[bool]]]

MIN_STEPS = 1
MAX_STEPS = 200

MIN_PRINT_STEP = 0  # print elapsed time for each solving step >= this value
#STEP_MAX_MINUTES = get(prg.get_const("step_max_mins"), Number(2)).number
#STEP_MAX_SECS = get(prg.get_const("step_max_secs"), Number(30)).number
STEP_MAX_MINUTES = 2
STEP_MAX_SECS = 30

STEP_MAX_ELAPSED_TIME = timedelta(minutes=STEP_MAX_MINUTES, seconds=STEP_MAX_SECS)

def get(val, default):
    return val if val != None else default

def tw_solve_incremental(prg: Control, initprg=None, imin=MIN_STEPS, imax=MAX_STEPS, istop="SAT"):

    _actions_list = []
    _actions_facts = []
    _newly_discovered_facts = []  # rooms or opened containers
    _recipe_read = False
    def _get_chosen_actions(model, step):
        #for act in prg.symbolic_atoms.by_signature("act",2):
        #     print(f"[t={act.symbol.arguments[1].number}] action:{act.symbol.arguments[0]}")
        nonlocal _actions_list
        nonlocal _actions_facts
        nonlocal _newly_discovered_facts
        print(f"_get_chosen_actions(model,{step})......")
        _actions_list.clear()
        _actions_facts.clear()
        _newly_discovered_facts.clear()
        _solved_all = False
        for atom in model.symbols(atoms=True):
            if (atom.name == "act" and len(atom.arguments)==2 and atom.arguments[1].type is SymbolType.Number):
                t = atom.arguments[1].number
                action = atom.arguments[0]
                assert action.arguments[0].number == t, f"!ACTION timestep mismatch: [{t}] {action}"
                str_atom = f"{atom}."
                _actions_facts.append(str_atom)
                print(f"  {'++' if step == t else '--'} [{t}] : {str_atom}")
                _actions_list.append((t, action))
            elif atom.name == 'recipe_read' \
              and len(atom.arguments)==1 and atom.arguments[0].type is SymbolType.Number:
                if atom.arguments[0].number == step:
                    print(f"  ++ {atom}")
                _newly_discovered_facts.append("cookbook")
            elif atom.name == 'solved_all' \
              and len(atom.arguments)==1 and atom.arguments[0].type is SymbolType.Number:
                print(f"  ++! {atom}")
                _newly_discovered_facts.append("solved_all")
                _solved_all = True
            elif (atom.name == 'first_visited' \
               or atom.name == 'first_opened'  \
               or atom.name == 'first_acquired') \
              and len(atom.arguments)==2 and atom.arguments[1].type is SymbolType.Number:
                if atom.arguments[1].number == step:    #, f"[{step}] {atom.name} {atom.arguments[1]}"
                    print(f"  ++ {atom}")
                    _newly_discovered_facts.append(str(atom.arguments[0]))
                else:
                    print(f"  -- {atom}")
        _actions_list = sorted(_actions_list, key=itemgetter(0))
        if _solved_all:
            return False
        return True  # False -> stop after first model

    step, ret = 0, None
    solved_all = False
    while ((imax is None or step < imax) and
           (step == 0 or step <= imin or (
              (istop == "SAT" and (not ret.satisfiable or not solved_all)
               or (istop == "UNSAT" and not ret.unsatisfiable)
               or (istop == "UNKNOWN" and not ret.unknown)
              )
           ))):
        start_time = datetime.now()
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
        if initprg:
            prg.add("initprg", [], initprg)
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
                actions_facts_str = "\n".join(_actions_facts)
                actions_facts_str = actions_facts_str.replace("act(", "did_act( ")
                print(f"\n+++++ ADDING prev_actions: +++\n{actions_facts_str}\n----", flush=True)
                #print("t", "\n\t".join(_actions_facts), flush=True)
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
        finish_time = datetime.now()
        elapsed_time = finish_time-start_time
        print("<< SATISFIABLE >>" if ret.satisfiable else "<< NOT satisfiable >>", flush=True)
        if step >= MIN_PRINT_STEP:
            print(f"--- [{step}] elapsed: {elapsed_time}")
        if elapsed_time > STEP_MAX_ELAPSED_TIME:
            print(f"--- [{step}] Step time {elapsed_time} > {STEP_MAX_ELAPSED_TIME} ... Stop solving.")
            break
        step = step+1
    if ret.satisfiable:
        return _actions_list, step
    else:
        return None, step


def part_str(part: Tuple[str, List[Symbol]]):
    '''
    Return a nice string representation of the given program part to ground.
    '''
    if part[1]:
        return f'{part[0]}({", ".join(map(str, part[1]))})'
    return f'{part[0]}'


def run(files: Sequence[str], initprg=None):
    ctl = Control()
    print('  loading files:')
    for file_ in files:
        print(f'  - {file_}')
        ctl.load(file_)
    if initprg:
        ctl.add(initprg)
    actions, iters = tw_solve_incremental(ctl, imin=MIN_STEPS, imax=MAX_STEPS, istop="SAT")
    if actions is not None:
        print(f"SOLVED! {files} ACTIONS =")
        #assert iters == len(actions)+1, f"{iters} {len(actions)}"
        commands = []
        for t, action in actions:
            print(f"[{t}] {str(action)}")
            commands.append(action)
        return commands
    else:
        print(f"FAILED TO SOLVE: steps={iters}, {files}")
        return []


def run_gamefile(filepath: str):
    from twutils.tw_asp import generate_ASP_for_game, tw_command_from_asp_action
    from textworld.generator import Game
    from pathlib import Path
    source_path = Path(__file__).resolve()
    source_dir = source_path.parent
    # print("SOURCE DIRECTORY:", source_dir)
    input_filename = filepath.replace(".ulx", ".json")
    input_filename = input_filename.replace(".z8", ".json")
    game = Game.load(input_filename)
    game_infos = game._infos 
    asp_for_game = generate_ASP_for_game(game, asp_file_path=None)
    actions = run([str(source_dir / 'tw_asp.lp')], initprg=asp_for_game)
    commands = []
    for action in actions:
        commands.append(tw_command_from_asp_action(action, game_infos))
    return commands

# print("-------------")
# print('Examaple 1:')
# run(['/Users/gstrazds/work2/github/gstrazds/TextWorld/tw_gamesNP/'+
# 'tw-cooking-recipe3+take3+cook+cut+open+go9-X3Y1sGOyca0phPxJ.lp'])

# print()
# print("-------------")
# print('Examaple 2:')

# run_gamefile('/Users/gstrazds/work2/github/gstrazds/TextWorld/tw_games/'+
# 'tw-cooking-recipe3+take3+cook+cut+open+go9-X3Y1sGOyca0phPxJ.z8')
