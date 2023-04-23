import json

from typing import List, Sequence, Tuple, Optional
from operator import itemgetter
from datetime import datetime, timedelta

from clingo.control import Control
from clingo import Function, String, Number, Symbol, SymbolType

Parts = Sequence[Tuple[str, List[Symbol]]]
Externals = Sequence[Tuple[Symbol, Optional[bool]]]

from twutils.tw_asp_incremental import tw_solve_incremental

def part_str(part: Tuple[str, List[Symbol]]):
    '''
    Return a nice string representation of the given program part to ground.
    '''
    if part[1]:
        return f'{part[0]}({", ".join(map(str, part[1]))})'
    return f'{part[0]}'


def run_asp_solver(files: Sequence[str], initprg=None, idstr=None):
    if idstr is None:
        idstr = str(files)
    ctl = Control()
    for file_ in files:
        print(f'#  + loading file: {file_}')
        ctl.load(file_)
    if initprg:
        ctl.add(initprg)
    actions, num_iters, step_times = tw_solve_incremental(ctl)  #, imin=MIN_STEPS, imax=MAX_STEPS, istop="SAT")
    if actions is not None:
        print(f"SOLVED! {idstr} ACTIONS =")
        #assert iters == len(actions)+1, f"{iters} {len(actions)}"
        commands = []
        for t, action in actions:
            print(f"[{t}] {str(action)} {str(timedelta(microseconds=step_times[t][0]))} SAT={str(step_times[t][1])}")
            commands.append(action)
        return commands, step_times
    else:
        print(f"FAILED TO SOLVE {idstr} steps={num_iters}, {files}, {step_times}")
        return [], step_times


def plan_commands_for_game(filepath: str):
    from twutils.tw_asp import generate_ASP_for_game, tw_command_from_asp_action
    from textworld.generator import Game
    from textworld.generator.game import EntityInfo
    from pathlib import Path
    source_path = Path(__file__).resolve()
    source_dir = source_path.parent
    # print("SOURCE DIRECTORY:", source_dir)
    asp_for_game = None
    # need to get game_infos from corresponding .json file
    input_filename = filepath.replace(".ulx", ".json")
    input_filename = input_filename.replace(".z8", ".json")
    input_filename = input_filename.replace(".0.lp", ".json")

    if filepath.endswith(".lp"):
        with open(input_filename, "r") as jsonfile:
            data = json.load(jsonfile)
        game_infos = {k: EntityInfo.deserialize(v) for k, v in data["infos"]}
        files = [filepath]
        if filepath.endswith(".0.lp"):  # was generated without shared rules
            files.append(str(source_dir / 'tw_asp.lp'))
    else: # not filepath.endswith(".lp"):
        game = Game.load(input_filename)
        game_infos = game._infos
        files = []
        print(f"## Converting {input_filename} to ASP")
        asp_for_game = generate_ASP_for_game(game,
            asp_file_path=None, standalone=True, emb_python=False)

    actions, step_times = run_asp_solver(files, initprg=asp_for_game, idstr=str(filepath))
    tw_commands = []
    for action in actions:
        tw_commands.append(tw_command_from_asp_action(action, game_infos))
    if filepath.endswith(".lp"):
        print(tw_commands)
    return tw_commands, step_times

# print("-------------")
# print('Examaple 1:')
# run(['/Users/gstrazds/work2/github/gstrazds/TextWorld/tw_gamesNP/'+
# 'tw-cooking-recipe3+take3+cook+cut+open+go9-X3Y1sGOyca0phPxJ.lp'])

# print()
# print("-------------")
# print('Examaple 2:')

# run_gamefile('/Users/gstrazds/work2/github/gstrazds/TextWorld/tw_games/'+
# 'tw-cooking-recipe3+take3+cook+cut+open+go9-X3Y1sGOyca0phPxJ.z8')
