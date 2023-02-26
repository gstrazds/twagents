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


def run(files: Sequence[str], initprg=None, idstr=None):
    if idstr is None:
        idstr = str(files)
    ctl = Control()
    for file_ in files:
        print(f'#  + loading file: {file_}')
        ctl.load(file_)
    if initprg:
        ctl.add(initprg)
    actions, iters = tw_solve_incremental(ctl)  #, imin=MIN_STEPS, imax=MAX_STEPS, istop="SAT")
    if actions is not None:
        print(f"SOLVED! {idstr} ACTIONS =")
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
    print(f"## Converting {input_filename} to ASP")
    game = Game.load(input_filename)
    game_infos = game._infos 
    asp_for_game = generate_ASP_for_game(game, asp_file_path=None)
    actions = run([str(source_dir / 'tw_asp.lp')], initprg=asp_for_game, idstr=str(filepath))
    tw_commands = []
    for action in actions:
        tw_commands.append(tw_command_from_asp_action(action, game_infos))
    return tw_commands

# print("-------------")
# print('Examaple 1:')
# run(['/Users/gstrazds/work2/github/gstrazds/TextWorld/tw_gamesNP/'+
# 'tw-cooking-recipe3+take3+cook+cut+open+go9-X3Y1sGOyca0phPxJ.lp'])

# print()
# print("-------------")
# print('Examaple 2:')

# run_gamefile('/Users/gstrazds/work2/github/gstrazds/TextWorld/tw_games/'+
# 'tw-cooking-recipe3+take3+cook+cut+open+go9-X3Y1sGOyca0phPxJ.z8')
