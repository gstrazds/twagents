from typing import List, Sequence, Tuple, Optional
from operator import itemgetter
from datetime import datetime, timedelta

from clingo.control import Control
from clingo import Function, String, Number, Symbol, SymbolType


_MIN_STEPS = 1
_MAX_STEPS = 200

_MIN_PRINT_STEP = 0  # print elapsed time for each solving step >= this value
_STEP_MAX_MINUTES = 9
_STEP_MAX_SECS = 30

_STEP_MAX_ELAPSED_TIME = timedelta(minutes=_STEP_MAX_MINUTES, seconds=_STEP_MAX_SECS)

def tw_solve_incremental( prg: Control, istop="SAT", imin=_MIN_STEPS, imax=_MAX_STEPS,
                          step_max_time=_STEP_MAX_ELAPSED_TIME, min_print_step=_MIN_PRINT_STEP):

    if min_print_step == -1:
        min_print_step = imax + 10000   # don't print step times

    _actions_list = []
    _actions_set = set()
    _new_actions = []
    _actions_facts = []
    _newly_discovered_facts = []  # rooms or opened containers
    _recipe_read = False
    def _get_chosen_actions(model, step):
        #for act in prg.symbolic_atoms.by_signature("act",2):
        #     print(f"[t={act.symbol.arguments[1].number}] action:{act.symbol.arguments[0]}")
        nonlocal _actions_list
        nonlocal _new_actions
        nonlocal _actions_facts
        nonlocal _newly_discovered_facts
        print(f"_get_chosen_actions(model,{step})......")
        _actions_list.clear()
        _actions_facts.clear()
        _new_actions.clear()
        _newly_discovered_facts.clear()
        _solved_all = False
        for atom in model.symbols(atoms=True):
            if (atom.name == "act" and len(atom.arguments)==2 and atom.arguments[1].type is SymbolType.Number):
                t = atom.arguments[1].number
                action = atom.arguments[0]
                assert action.arguments[0].number == t, f"!ACTION timestep mismatch: [{t}] {action}"
                str_atom = f"{atom}."
                _actions_facts.append(str_atom)
                # print(f"  {'++' if step == t else '--'} [{t}] : {str_atom}")
                _actions_list.append((t, action))
                if action not in _actions_set:
                    print(f"\t ++++ new_act: [{t}] : {str(action)} ++++")
                    _actions_set.add(action)
                    _new_actions.append(action)

            elif atom.name == 'goal1_achieved' \
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
        for t, action in _actions_list:
            # if action not in _actions_set:
            #     print(f"\t ++++ new_act: [{t}] : {str(action)} ++++")
            #     _actions_set.add(action)
            #     _new_actions.append(action)
            print(f"  {'++' if step == t else '--'} [{t}] : {action}")
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
        if step >= min_print_step:
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

            for action in _new_actions:
                print("ENSURE PAST ACTION:", str(action))
                parts.append(("prev_action", [action, action.arguments[0]]))
                # prg.assign_external(Function("did_act", [action, action.arguments[0]]), True)
            _new_actions.clear()  # only need to add these actions once
            # if len(_actions_facts):
            #     actions_facts_str = "\n".join(_actions_facts)
            #     actions_facts_str = actions_facts_str.replace("act(", "did_act( ")
            #     print(f"\n+++++ ADDING prev_actions: +++\n{actions_facts_str}\n----", flush=True)
            #     print("\t" + "\n\t".join(_actions_facts), flush=True)
            #     prg.add("prev_actions", [], actions_facts_str)
            #     parts.append(("prev_actions", []))

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
        assert len(_actions_set) == len(_actions_list), f"@@ _actions_set {len(_actions_set)}: {_actions_set} \n@@ _actions_list {len(_actions_list)}: {_actions_list}"

        if step >= min_print_step:
            print(f"--- [{step}] elapsed: {elapsed_time}")
        if elapsed_time > step_max_time:
            print(f"--- [{step}] Step time {elapsed_time} > {step_max_time} ... Stop solving.")
            break
        step = step+1
    if ret.satisfiable:
        return _actions_list, step
    else:
        return None, step
