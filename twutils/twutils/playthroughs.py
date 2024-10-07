import json
from collections import defaultdict, OrderedDict
from datetime import timedelta
import os
import os.path
import glob
from pathlib import Path

from typing import List, Dict, Mapping, Optional, Any

import textworld
import textworld.gym
from textworld.logic import Proposition  #, Variable, Signature, State

from .file_helpers import count_iter_items, split_gamename  # , parse_gameid
from .twlogic import filter_observables, subst_names
from .feedback_utils import normalize_feedback_vs_obs_description, simplify_feedback, INSTRUCTIONS_TOKEN
from symbolic.format_desc import END_OF_LIST

from symbolic.knowledge_graph import KnowledgeGraph
from symbolic.wrappers.oracle_wrappers import TWoWrapper, TwAspWrapper


# default directory paths, usually overriden by env, config or cmd line args
TW_GAMES_BASEDIR = '/work2/gstrazds/twdata/ftwc/games_ftwc/'
TW_TRAINING_DIR = TW_GAMES_BASEDIR + 'train/'
DEFAULT_FTWC_PTHRU_BASE = '/work2/gstrazds/twdata/ftwc/playthru_data/'
DEFAULT_GATA_PTHRU_BASE = '/work2/gstrazds/twdata/gata/playthru_data/'

# TW_VALIDATION_DIR = TW_GAMES_BASEDIR + 'valid/'
# TW_TEST_DIR = TW_GAMES_BASEDIR + 'test/'

CMD_START_TOKEN = '>>>['
CMD_END_TOKEN = ']<<<'
PATHTRACE_TOKEN = 'pathtrace'
GAME_START_CMD = 'start'
JSON_LINE_SEP = ' '         #' <|> '
JSON_CRLF = ' <|> '   # replaces '\n'* when merging raw text output into JSON for .textds dataset file

DEFAULT_PTHRU_SEED = 42
MAX_PLAYTHROUGH_STEPS = 150

GOAL_MEAL = 'eatmeal'
GOAL_ANSWER_WHERE = 'whereis'
GOAL_ANSWER_EXISTS = 'exists'


def normalize_path(path_str: str, subdir: str = None):
    if subdir:
        path_str = os.path.join(path_str, subdir)
    path_str = os.path.normpath(path_str)
    path_str = os.path.join(path_str, '')   # make sure it ends with a trailing slash
    return path_str


def make_dsfilepath(dirpath, dataset_name:str) -> str:
    return normalize_path(dirpath) + f'../{dataset_name}.textds'


def get_games_dir(basepath: str = None, splitname: str = 'train'):
    if not basepath:
        basepath = os.getenv('TW_GAMES_BASEDIR', TW_GAMES_BASEDIR)
    games_dir = basepath
    return normalize_path(games_dir, splitname)


def playthrough_id(objective_name=None, seed=None):
    if seed is None:
        seed = DEFAULT_PTHRU_SEED
    if not objective_name:
        objective_name = GOAL_MEAL

    return f"{objective_name}_{seed}"


class GamesIndex:
    def __init__(self):
        self._index = {}   # maps game_name => {'skills': list of skills, 'dir': index in _game_dirs}
        self.game_dirs = []
        self.pthru_dirs = []
        self.skills_index = {}    # maps skill name => set of gamenames
        self._skills_up_to_date = False

    def recompute_skills_index(self):
        self.skills_index = defaultdict(set)
        for gn in self._index:
            skills_list = self.get_skills_for_game(gn)
            if not skills_list:
                _gid_, skills_list = split_gamename(gn)
            if skills_list:
                for skill in skills_list:
                    self.skills_index[skill].add(gn)
        self._skills_up_to_date = True

    def get_skills_map(self):
        if not self._skills_up_to_date:
            self.recompute_skills_index()
        return self.skills_index

    def add_games_to_index(self, games_dir: str, game_names: List[str]):
        if not game_names:
            print("WARNING( GamesIndex.add_games_to_index called with EMPTY list of game_names:", game_names)
            return
        try:
            dir_index = self.game_dirs.index(games_dir)
        except ValueError:
            self.game_dirs.append(games_dir)
            dir_index = len(self.game_dirs)-1

        n_dupl = 0
        n_added = 0
        for gn in game_names:
            gid, skills = split_gamename(gn)
            if gn in self._index:
                n_dupl += 1
                skills0 = self._index[gn].get('skills', None)
                gid0 = self._index[gn].get('gid', None)
                idx0 = self._index[gn].get('dir', None)
                assert skills0 == skills, f"Inconsistent existing entry for {games_dir} / {gn}: {skills0} {skills}"
                assert gid0 == gid, f"Inconsistent existing entry for {games_dir} / {gn}: {gid0} {gid}"
                assert idx0 == dir_index, f"Inconsistent existing entry for {games_dir} / {gn}: {idx0} {dir_index}"
            else:
                self._index[gn] = {'dir': dir_index, 'skills': skills, 'gid': gid}
                n_added += 1
        print(f"Added {n_added} games ; {n_dupl} duplicates skipped")

    def add_pthrus_to_index(self, pthrus_dir: str, game_names: List[str]):
        if not game_names:
            print("WARNING( GamesIndex.add_pthrus_to_index called with EMPTY list of game_names:", game_names)
            return
        try:
            dir_index = self.pthru_dirs.index(pthrus_dir)
        except ValueError:
            self.pthru_dirs.append(pthrus_dir)
            dir_index = len(self.pthru_dirs)-1

        n_dupl = 0
        n_added = 0
        n_updated = 0
        for gn in game_names:
            if gn not in self._index:
                self._index[gn] = {'ptdir': dir_index}
                n_added += 1
            else:
                assert gn in self._index, f"Game {gn} must be indexed before playthough can be indexed"
                if 'ptdir'in self._index[gn]:
                    if self._index[gn]['ptdir'] != dir_index:
                        n_updated += 1
                        print(f"WARNING: replacing {self.get_dir_for_pthru(gn)} <= {pthrus_dir}")
                    else:
                        n_dupl += 1
                else:
                    n_added += 1
                self._index[gn]['ptdir'] = dir_index

        print(f"Added {n_added} pthtrus + {n_updated}  updated  + {n_dupl} unchanged")

    def get_dir_for_game(self, game_name):
        if not game_name in self._index:
            return None
        return self.game_dirs[self._index[game_name]['dir']]

    def get_dir_for_pthru(self, game_name):
        if not game_name in self._index:
            return None
        idx_ptdir = self._index[game_name].get('ptdir', None)
        if idx_ptdir is None:
            return None
        return self.pthru_dirs[idx_ptdir]

    def get_skills_for_game(self, game_name):
        if not game_name in self._index:
            return None
        return self._index[game_name]['skills']

    def count_and_index_gamefiles(self, which, dirpath, suffixes=None):
        if not suffixes:
            suffixes = ['.ulx', '.z8', ]  #'.json']
        game_names_ = []
        all_files = os.listdir(dirpath)
        # directory contains up to 3 files per game: *.json, *.ulx, *.z8
        print(f"Total files in {dirpath} = {count_iter_items(all_files)}" )
        for suffix in suffixes:
            game_files = list(filter(lambda fname: fname.endswith(suffix), all_files))
            if game_files:
                print("number of {} files in {} = {}".format(suffix, dirpath, len(game_files)))
                game_names_ = [s.split('.')[0] for s in game_files]
        self.add_games_to_index(dirpath, game_names_)
        return game_names_

    def get_gamefile(self, game_name, suffix=None):
        if not suffix:
            suffixes = ['.z8', '.ulx', ]  #'.json']
        else:
            suffixes = [suffix]
        gamedir = self.get_dir_for_game(game_name)
        if not gamedir:
            assert False, f"Directory is not indexed for game {game_name}"
            return None
        for suffix in suffixes:
            if os.path.exists(os.path.join(gamedir, game_name+suffix)):
                gamefile = os.path.join(gamedir, game_name+suffix)
                return gamefile
        return None

    def count_and_index_pthrus(self, which, dirpath):
        suffixes = ['.pthru']
        game_names_ = []
        all_files = os.listdir(dirpath)
        # directory contains up to 3 files per game: *.json, *.ulx, *.z8
        print(f"Total files in {dirpath} = {count_iter_items(all_files)}" )
        for suffix in suffixes:
            game_files = list(filter(lambda fname: fname.endswith(suffix), all_files))
            if game_files:
                print("number of {} files in {} = {}".format(suffix, dirpath, len(game_files)))
                game_names_ = [s.split('.')[0] for s in game_files]
        self.add_pthrus_to_index(dirpath, game_names_)
        return game_names_

# def parse_gameid(game_name: str) -> str:
#     game_id = game_name[11:] if game_name.startswith("tw-cooking-") else game_name
#     segments = game_id.split('-')
#     if len(segments) >= 2:
#         code, guid = segments[0:2]
#         guid = guid.split('.')[0]
#         guid = "{}..{}".format(guid[0:4],guid[-4:])
#         segments = code.split('+')
#         r, t, g, k, c, o, d = ('0', '0', 0, '*', '*', '*', '*')
#         for seg in segments:
#             if seg.startswith('recipe'):
#                 r = seg[len('recipe'):]
#             elif seg.startswith('go'):
#                 g = int(seg[len('go'):])
#             elif seg.startswith('take'):
#                 t = seg[len('take'):]
#             elif seg == 'cook':
#                 k = 'k'
#             elif seg == 'cut':
#                 c = 'c'
#             elif seg == 'open':
#                 o = 'o'
#             elif seg == 'drop':
#                 d = 'd'
#             else:
#                 assert False, "unparsable game_id: {}".format(game_id)
#         shortcode = "r{}t{}{}{}{}{}g{:02d}-{}".format(r,t,k,c,o,d,g,guid)
#     else:
#         shortcode = game_id
#     return shortcode


def _collect_gamenames(games_dir=None):
    if not games_dir:
        games_dir = get_games_dir()    # retrieve the default dir for training games (based on env and config)

    training_files = os.listdir(games_dir)
    # directory contains 3 files per game: *.json, *.ulx, *.z8
    print("Total training files = ", count_iter_items(training_files))
    suffixes = ['.json', '.z8', '.ulx']
    for suffix in suffixes:
        training_games = list(filter(lambda fname: fname.endswith(suffix), training_files))
        print("number of {} files in {} = {}".format(suffix, games_dir, len(training_games)))
    game_names_ = [s.split('.')[0] for s in training_games]

    if game_names_ and game_names_[0].startswith('tw-cooking-'):
        # remove the 'tw-cooking-' prefix
        game_names = list(map(lambda gn: gn[11:], game_names_))
        games = defaultdict(list)
        for gn in game_names:
            # skills, gid = gn.split('-')
            gid, skills = split_gamename(gn)
            if not skills or not skills[0]:
                print("SPLIT FAILURE:", gn)
            else:
                games[gid].append(gn) #defaultdict initializes with empty list

        print("Number of training games", len(training_games))
        print("Number of unique IDs", len(games))
        print()
        total = 0
        for n in [1, 2, 3, 4, 5]:
            c = count_iter_items(filter(lambda g: len(g) == n, games.values()))
            print("IDs with {} entries: {}".format(n, c))
            total += n * c
        print("Total =", total)
        assert total == len(game_names)
    else:
        game_names = game_names_
    return game_names


cooking_adjs = ['raw', 'diced', 'fried', 'chopped', 'grilled',
                'baked', 'broiled', 'boiled', 'roasted', 'toasted']

#remove (closed), (locked), (open) if 
def remove_parenthetical_state(line):
    splitline = line.split(' (')
    if len(splitline) > 1 and line[-1] == ')':
        return splitline[0]
    #else:
    return line

# removes "a|an|some" and also decreases indentation by 2 spaces
def remove_prewords(item):
#     print("remove_prewords <{}> -> ".format(item), end='')
    item2 = item.strip()
    indent = item.find(item2)
    out_indent = 0
    if indent >= 2:
        out_indent = indent - 2  # decrease indentation
    if item2.startswith("a "):
         item2= item2[2:]
    elif item2.startswith("an "):
        item2 = item2[3:]
    elif item2.startswith("some "):
        item2 = item2[5:]
    item2 = item2.rjust(len(item2)+out_indent)
#     print("<{}>".format(item2))
    return item2, indent


def start_twenv(gamefile,
                # max_episode_steps=MAX_PLAYTHROUGH_STEPS,
                # random_seed=DEFAULT_PTHRU_SEED,
                pthru_cmds=None,
                step_infos=None,
                env_infos=None,
                ):
    if not env_infos:
        env_infos = textworld.EnvInfos(game=True, facts=True, feedback=True, description=True, inventory=True, location=True,
                                   last_action=True, last_command=True, intermediate_reward=True)
    twenv = textworld.start(gamefile, wrappers=[TwAspWrapper], request_infos=env_infos)
    twenv.pthru_cmds = pthru_cmds
    twenv._planner_step_times = step_infos
    # even if use_internal_names is True, currently works only if the oracle internally uses human readable names
    # (names get remapped to internal ids in export_playthru( remap_names=names2ids )
    twenv.use_internal_names = False   #use_internal_names
    game_state = twenv.reset()
    # game_state = twenv.get_initial_state()
    print(game_state.keys())
    reward = game_state.reward
    done = False
    start_cmd = 'start'
    step_time = (0,True)
    ## names2ids = twenv.tw_oracle.map_names2ids if use_internal_names else None
    #obs, rewards, dones, infos = gather_infos_for_playthroughs([game_state], rewards, dones, start_cmds, step_times)
    feedback, infos = gather_twstep_info(game_state, done, start_cmd, step_time)  #,names2ids
    return twenv, feedback, infos

def step_twenv(twenv, cmd_str:str):
    if not cmd_str:
        print("WARNING - step_twenv_for_playthrough: EMPTY command! substituting 'do nothing'")
        cmd = 'do nothing'
    else:
        cmd = cmd_str
    game_state, reward, done = twenv.step(cmd)
    # names2ids = twenv.tw_oracle.map_names2ids if export_internal_names else None
    step_time = twenv.get_twenv_step_time_info()
    if step_time is None:
        step_time = (0,True)
    feedback, infos = gather_twstep_info(game_state, done, cmd, step_time)  #,names2ids
    return game_state, feedback, reward, done, infos, step_time


def gather_twstep_info(gs: textworld.GameState,
                                  done: bool,
                                  cmd: str,
                                  step_time: Optional[tuple],  #Tuple
                                  ):
    infos = {}
    if done and gs.last_command == 'do nothing':   # stop appending to infos for finished games
        feedback = None
    else:
        feedback = gs.feedback if gs.feedback else ''
        # ------- standard TextWorld
        infos['game'] = gs.game
        infos['facts'] = gs.facts
        infos['feedback'] = gs.feedback if gs.feedback else ''
        infos['description'] = gs.description if gs.description else ''
        infos['inventory'] = gs.inventory if gs.inventory else ''
        infos['prev_action'] = gs.last_command
        infos['admissible_commands'] = gs.admissible_commands   # sorted(set(gs["_valid_commands"])))
        # ------- custom from TWoWrapper
        infos['reward'] = gs.reward
        infos['game_score'] = gs.score
        infos['tw_o_step'] = gs.next_command
        infos['done'] = done
        if hasattr(gs, '_tasks'):
            infos['tw_o_stack'] = gs._tasks
        if step_time is not None:
            infos['solver_step_time'] = step_time[0]
            infos['solver_sat'] = step_time[1]
    return feedback, infos

def start_twenv_for_playthrough(gamefiles,
                                max_episode_steps=MAX_PLAYTHROUGH_STEPS,
                                random_seed=DEFAULT_PTHRU_SEED,
                                pthru_cmds=None,
                                step_infos=None,
                                ):
    env_infos = textworld.EnvInfos(game=True, facts=True, feedback=True, description=True, inventory=True, location=True,
                                   last_action=True, last_command=True, intermediate_reward=True)
    batch_size = len(gamefiles)
    assert batch_size == 1, f"Currently only support batch_size=1 (not:{len(gamefiles)} {gamefiles})"
    twenv = textworld.start(gamefiles[0], wrappers=[TwAspWrapper], request_infos=env_infos)
    twenv.pthru_cmds = pthru_cmds
    twenv._planner_step_times = step_infos
    # even if use_internal_names is True, currently works only if the oracle internally uses human readable names
    # (names get remapped to internal ids in export_playthru( remap_names=names2ids )
    twenv.use_internal_names = False   #use_internal_names
    game_state = twenv.reset()
    # game_state = twenv.get_initial_state()
    print(game_state.keys())
    game_states = [game_state]
    rewards = [gs.reward for gs in game_states]
    dones = [False] * batch_size
    start_cmds = ['start'] * batch_size
    step_times = [(0,True) for _ in range(batch_size)]
    ## names2ids = twenv.tw_oracle.map_names2ids if use_internal_names else None
    #obs, rewards, dones, infos = gather_infos_for_playthroughs([game_state], rewards, dones, start_cmds, step_times)
    feedback, infos = gather_twstep_info(game_state, dones[0], start_cmds[0], step_times[0])  #,names2ids
    feedbacks = [feedback]
    all_infos = merge_infos_for_playthroughs([infos])  #,names2ids
    return twenv, feedbacks, all_infos

def step_twenv_for_playthrough(twenv, step_cmds:List[str]):
    assert len(step_cmds) == 1, f"Currently, only support batch_size=1 {step_cmds}"
    game_state, feedback, reward, done, infos, step_time = step_twenv(twenv, step_cmds[0])
    # game_states = [game_state]
    feedbacks = [feedback]
    rewards = [reward]
    dones = [done]
   # names2ids = twenv.tw_oracle.map_names2ids if export_internal_names else None
    all_infos = merge_infos_for_playthroughs([infos])  #,names2ids
    return feedbacks, rewards, dones, all_infos

def merge_infos_for_playthroughs(array_of_infos):
    required_info_keys = [
        # ------- standard TextWorld
         'game',
        'facts',
        'feedback',
        'description',
        'inventory',
        'prev_action',
        'admissible_commands',
        # ------- custom from TWoWrapper
        'reward',
        'game_score',
        'done',
        #'tw_o_step',
        #'solver_step_time',
        #'solver_sat',
    ]
    all_infos = {key: [] for key in array_of_infos[0].keys()}
    for idx, infos in enumerate(array_of_infos):
        for key in required_info_keys:
            assert key in infos, f"key '{key}' missing from array_of_infos[{idx}]: {infos.keys()}"
        for key in infos.keys():
            all_infos[key].append(infos[key])
    return all_infos


def extract_pathtrace(cmd_history):
    pathtrace = []
    for (cmd, locname, ok, _reward_) in reversed(cmd_history):
        if ok and cmd.startswith("go "):
            pathtrace.append((cmd, locname))
        else:
            break
    pathtrace.reverse()
    return pathtrace

def playthrough_step_to_json(cmds: List[str],
                             dones: List[bool],
                             infos, # a dictionary of lists
                             obs: List[Optional[str]],
                             rewards: List[float],
                             step_num: int,
                             env=None,
                             pathtrace=False,
                             ) -> List[Any]:
    if pathtrace:
        assert env is not None, "Must pass a TWoWrapper wrapped env for playthough_step_to_json(pathtrace=True)"
        assert hasattr(env, "tw_oracle")
        assert len(dones) == 1   # at least for now, this bit of code assumes batch size = 1
        cmd_histories = [env.tw_oracle.cmd_history.copy()]  # .copy()
    else:
        cmd_histories = None
    step_json_list = []
    for idx in range(len(dones)):
        world_facts = infos['facts'][idx]
        observable_facts, player_room = filter_observables(world_facts, game=infos['game'][idx])
        world_facts_serialized = [f.serialize() for f in world_facts]
        observable_facts_serialized = [f.serialize() for f in observable_facts]
        # step_key = format_stepkey(step_num)
        oracle_action = infos['tw_o_step'][idx] if 'tw_o_step' in infos else None
        oracle_stack = infos['tw_o_stack'][idx] if 'tw_o_stack' in infos else "(( )) [[ ]]"
        solver_step_time = infos['solver_step_time'][idx] if 'solver_step_time' in infos else None
        solver_sat = infos['solver_sat'][idx] if 'solver_sat' in infos else None
        cmd_history = cmd_histories[idx] if cmd_histories else None
        step_json = {
            # step_key: {
                'reward': rewards[idx],
                'score': infos['game_score'][idx],
                'done': dones[idx],
                'player_location': player_room.name,
                'obs': obs[idx],
                'feedback': infos['feedback'][idx],
                'description': infos['description'][idx],
                'inventory': infos['inventory'][idx],
                'prev_action': cmds[idx],
                'next_action': oracle_action,
                'possible_actions': infos['admissible_commands'][idx],
                'obs_facts': observable_facts_serialized,
                'GT_FACTS': world_facts_serialized,
            # }
        }
        if oracle_stack:
            # step_json[step_key]['tw_o_stack'] = oracle_stack
            step_json['tw_o_stack'] = oracle_stack
        if solver_step_time is not None:
            step_json['solver_step_time'] = solver_step_time
        if solver_sat is not None:
            step_json['solver_sat'] = solver_sat
        if cmd_history is not None:
            step_json['pathtrace'] = extract_pathtrace(cmd_history)

        # if step_num == 0 or dones[0]:  # at start of game or game over
        #     step_json[step_key]['GT_FACTS'] = world_facts_serialized
        step_json_list.append(step_json)
    return step_json_list


def generate_playthrus(gamefiles: List[str], randseed=DEFAULT_PTHRU_SEED, with_pathtrace=False):
    def append_step_data(array_list: List[List[Any]], step_list: List[Any]):
        assert len(array_list) == len(step_list), f"Batch size should match {len(array_list)} {len(step_list)}"
        for idx, step_data in enumerate(step_list):   # an entry for each game in the batch (can be None if game is_done)
            if step_data:                             # games that have finished will have None
                array_list[idx].append(step_data)     # result is ragged: differing length lists of step data for each game

    batch_size = len(gamefiles)
    num_steps = 0
    _dones = [0] * batch_size
    _rewards = [0] * batch_size
    next_cmds = ['start'] * batch_size
    env, _obs, _infs = start_twenv_for_playthrough(gamefiles, random_seed=randseed)
    playthru_step_data = playthrough_step_to_json(next_cmds, _dones, _infs, _obs, _rewards,
                                                  num_steps, env=env, pathtrace=with_pathtrace)
    # playthru_step_data is a list of list of json dicts (with data for a single game step),
    #   one entry for each game in the batch
    step_array_list = [playthru_step_data]  # Expecting that every game will always have at least one initial step
    #step_array_list: List[List[Any]]  # a list of lists: of step info from playing a single game

    next_cmds = _infs['tw_o_step']
    while not all(_dones) and num_steps < MAX_PLAYTHROUGH_STEPS+1:
        num_steps += 1
        # if _dones[0]:
        #     game_over += 1  # invoke one extra step after the last real step
        _obs, _rewards, _dones, _infs = step_twenv_for_playthrough(env, next_cmds)
        playthru_step_data = playthrough_step_to_json(next_cmds, _dones, _infs, _obs, _rewards,
                                                      num_steps, env=env, pathtrace=with_pathtrace)
        append_step_data(step_array_list, playthru_step_data)
        next_cmds = _infs['tw_o_step']
    env.close()
    return step_array_list, [env.tw_oracle.get_game_data()]


def format_facts(facts_list, prev_action=None, obs_descr=None, kg=None):
    if not isinstance(facts_list[0], Proposition):   # maybe serialized list of facts
        # world_facts0 = world_facts.copy()
        facts_list = [Proposition.deserialize(fact_json) for fact_json in facts_list]

    if kg is None:
        kg = KnowledgeGraph(debug=False)   # suppress excessive print() outs
    obs_descr = kg.update_facts(obs_descr, facts_list, prev_action=prev_action)
    #return str(kg)
    return kg.describe_room(kg.player_location.name, obs_descr=obs_descr)


def get_kg_descr(kg_accum, stepdata):
    # obs_facts_key = 'oservable_facts' if 'oservable_facts' in stepdata else 'obs_facts'  # TODO: delete this line
    prev_action = stepdata['prev_action']
    observed_facts = stepdata['obs_facts']
    # print( "---==================================================================---")
    # print(f"[{i:02d}] ++++ LOCATION:", stepdata['player_location'])
    # print(f"     ++++ DID:",      prev_action)
    # print(f"     ++++ FEEDBACK:", stepdata['feedback'])
    # print(f" ++++++++ OBS +++++++++++++++++++++++++++++++++++++++++:\n{stepdata['obs']}")
    # print( " ......................................................")
    # print(f"     ++++ DESCR:", stepdata['description'])
    # print(f"     ++++ CARRYING:", stepdata['inventory'])
    # print(f"     ---- CAN TRY:",  stepdata['possible_actions'])
    # print(f"     >>>> NEXT_ACT:", stepdata['next_action'])
    # #print(f"     **** FACTS:",    stepdata[obs_facts_key])
    # print( "========= KG_VISIBLE ====================================================")
    # print(stepdata.keys())
    # print(stepdata['description'])
    kg_descr = format_facts(observed_facts, kg=kg_accum, prev_action=prev_action, obs_descr=stepdata['obs'])
    return kg_descr


def format_playthrough_step(kg_descr, stepdata, simplify_raw_obs_feedback=True, no_kg_info=False):
    feedback = stepdata['feedback']
    prev_action = stepdata['prev_action']
    # for k in stepdata.keys():
    #     print(f"\t{k}: {stepdata[k]}")
    if simplify_raw_obs_feedback:
        new_feedback = normalize_feedback_vs_obs_description(prev_action,
                                                             stepdata['obs'],
                                                             stepdata['feedback'],
                                                             stepdata['description'])
        if new_feedback:
            print(f"format_playthrough_step MODIFYING ['feedback'] : '{new_feedback}' <-- orig:", stepdata['feedback'])
            feedback = new_feedback

   # print(f"[{i}] {gn} .....")
    cmdstr = f"\n{CMD_START_TOKEN} {prev_action} {CMD_END_TOKEN}\n"
    # if 'rtg' in stepdata:
    #     pthru += f"[[[ {stepdata['rtg']} ]]]\n"
    # if 'tw_o_stack' in stepdata:
    #     pthru += stepdata['tw_o_stack']
    outstr = cmdstr
    pthru = cmdstr  #_pthru_ + cmdstr
    # pthru_out += outstr
    if feedback: #and prev_action != 'start':
        outstr += feedback
        # feedback = feedback.strip()
    pthru += simplify_feedback(feedback) + '\n'
    pthru += kg_descr + '\n'
    if not no_kg_info and 'pathtrace' in stepdata:
        pathtrace_data = stepdata['pathtrace']
        if len(pathtrace_data):
            pathtrace = []
            for pathstep in pathtrace_data:
                pathtrace.append(pathstep[1])  # location name
                pathtrace.append(pathstep[0][3:]) # go <direction>
            pthru += f"{PATHTRACE_TOKEN} {' '.join(pathtrace)} {END_OF_LIST}\n"
    # outstr += '\n' + stepdata['obs']
    outstr += '\n' + stepdata['description'] + '\n'
    if 'inventory' in stepdata and stepdata['inventory']:
        outstr += '\n' + stepdata['inventory'] + '\n'
    return outstr, pthru


def concat_pthru_step(pthru_so_far:str, pthru_step:str, keep_objectives=True, is_last=False) -> str:
    if keep_objectives:
        instr = []
        lines_so_far = pthru_so_far.split('\n')  # parse pthru_so_far
        lines_step = pthru_step.split('\n')
        while not lines_step[-1].strip():
            lines_step = lines_step[:-1]   # remove empty lines from the end
        lines_so_far.extend(lines_step)
        for line in lines_so_far:
            if INSTRUCTIONS_TOKEN in line:
                instr = line[line.index(INSTRUCTIONS_TOKEN):]  # copy from token to end of line
                # keep iterating, if more than one line matches, we keep only the last one ...
        if instr and not lines_step[-1].startswith(INSTRUCTIONS_TOKEN):  # if step already ends with instr, don't double it
            if is_last:  # if this is the end of the game, change the descr of objectives
                pthru_step = pthru_step + f"{INSTRUCTIONS_TOKEN} GAME_OVER ;\n"
            else:
                pthru_step = pthru_step + instr + '\n'  # (put back the EOL that we stripped off from instr line)
    return pthru_so_far + pthru_step


def _list_game_files(dirpath):
    game_names_ = []
    all_files = os.listdir(dirpath)
    # directory contains up to 3 files per game: *.json, *.ulx, *.z8
    print(f"Total files in {dirpath} = {count_iter_items(all_files)}" )
    suffixes = ['.ulx', '.z8', '.json']  # will list all suffixes, but return only the last list
    for suffix in suffixes:
        game_files = list(filter(lambda fname: fname.endswith(suffix), all_files))
        if game_files:
            print("number of {} files in {} = {}".format(suffix, dirpath, len(game_files)))
            game_names_ = [s.split('.')[0] for s in game_files]   # return only the last of 3 possible lists
    return game_names_


def format_rtg_for_json(playthru, rtg=True):
    if 'rtg' in playthru[0]:
        rtg_list_str = ','.join([str(stepdata['rtg']) for stepdata in playthru])
        return f"[ {rtg_list_str} ]"
    return ''


def format_taskstacks_for_json(playthru):
    # def _format_taskstack(stackstr):
    #     return stackstr
    if 'tw_o_stack' in playthru[0]:
        tasks_list = [stepdata['tw_o_stack'] for stepdata in playthru]
        return json.dumps(tasks_list)
    return ''


def format_solver_info_json(playthru):
    if 'solver_step_time' in playthru[0]:
        assert 'solver_sat' in playthru[0]
        solver_info_list = [{'solver_time': stepdata['solver_step_time'], 'solver_sat':stepdata['solver_sat']} for stepdata in playthru]
        return json.dumps(solver_info_list)
    return '[]'


def export_playthru(gn, playthru, destdir='.', dry_run=False, rtg=True,
                    map_names2ids: Optional[Mapping[str,str]] = None, dataset_name=None):

    # gn = 'tw-cooking-recipe3+take3+cut+go6-Z7L8CvEPsO53iKDg'
    # gn = 'tw-cooking-recipe1+cook+cut+open+drop+go6-xEKyIJpqua0Gsm0q'
    #_rj = Client(host='localhost', port=6379, decode_responses=True)

    # ! NB: the following code modifies the contents of the retrieved playthru json in memory (but not in Redis)
    pthru_all = []
    othru_all = []   # similar, but without using any knowledge from previous observations
    raw_all = []
    tstacks_all = []

    export_internal_names = True if map_names2ids else False
    kg_accum = KnowledgeGraph(names2ids=map_names2ids,
                              use_internal_names=export_internal_names,
                              debug=True)   # debug=False suppresses excessive print()
    num_files = 0
    max_score = max([stepdata['score'] for stepdata in playthru])
    end_score = playthru[-1]['score']
    # print(f"export_playthru num_steps:{len(playthru)} max_score:{max_score} end_score:{end_score}")
    return_to_go = end_score  # max_score
    for i, stepdata in enumerate(playthru):
        is_last_step = (i >= len(playthru)-1)
        prev_action = stepdata['prev_action']
        if rtg:
            stepdata['rtg'] = return_to_go
        # print(f"export_playthru return_to_go={return_to_go} reward={stepdata['reward']}")
        return_to_go -= stepdata['reward']
        # print( "export_playthru:", return_to_go == max_score - stepdata['score'], f"max_score:{max_score} score:{stepdata['score']} rtg:{return_to_go}")
        assert return_to_go == max_score - stepdata['score'], f"max_score:{max_score} score:{stepdata['score']} rtg:{return_to_go}"
        if prev_action and " the " in prev_action:
            prev_action = prev_action.replace(" the ", " ")
            stepdata['prev_action'] = prev_action

        kg_descr = get_kg_descr(kg_accum, stepdata)
        prev_options = kg_accum.set_formatting_options('parsed-obs')
        assert prev_options.startswith('kg-descr'), prev_options
        kg_descr_without_oracle = get_kg_descr(kg_accum, stepdata)
        kg_accum.set_formatting_options(prev_options)

        # because saved playthroughs have raw feedback and obs, do what the ConsistentFeedbackWrapper would normally do
        _, pthru0 = format_playthrough_step(kg_descr_without_oracle, stepdata, simplify_raw_obs_feedback=True, no_kg_info=True)
        if map_names2ids:
            pthru0 = subst_names(pthru0, map_names2ids)
        taskstack = stepdata['tw_o_stack'] if 'tw_o_stack' in stepdata else ''
        if map_names2ids:
            taskstack = subst_names(taskstack, map_names2ids)
        tstacks_all.append(taskstack)

        outstr, pthru = format_playthrough_step(kg_descr, stepdata, simplify_raw_obs_feedback=True)
        if map_names2ids:
            pthru = subst_names(pthru, map_names2ids)
            outstr = subst_names(outstr, map_names2ids)
        pthru_all.append(pthru)
        othru_all.append(pthru0)
        raw_all.append(outstr)

        xdir = destdir + '/' + gn
        if not os.path.exists(xdir):
            if not dry_run:
                os.makedirs(xdir)
        if not dry_run:
            with open(xdir+f'/_step_{i:02d}.txt', 'w') as outfile:
                outfile.write(outstr)
        num_files += 1
        if not dry_run:
            with open(xdir + f'/step_{i:02d}.pthru', 'w') as outfile:
                outfile.write(pthru)
                #outfile.write(taskstack+pthru)
        num_files += 1
        if not dry_run:
            with open(xdir + f'/step_{i:02d}.othru', 'w') as outfile:
                outfile.write(pthru0)
        num_files += 1
    if not dry_run:
        # output the full playthrough
        accum_pthru = ''
        for i, (pthru, taskstack) in enumerate(zip(pthru_all, tstacks_all)):
            is_last_step = (i >= len(pthru_all) - 1)
            if False:   # here is where we can add taskstack to pthru records (but it's wrong for new ASP oracle, and benefits aren't well established even otherwise)
                pthru_step_ = (taskstack + pthru)
            else:
                pthru_step_ = pthru  # temporarily don't output taskstack (for comparison)
            accum_pthru = concat_pthru_step(accum_pthru, pthru_step_, keep_objectives=True, is_last=is_last_step)

        accum_othru = "\n".join(othru_all)
        with open(destdir+f'/{gn}.pthru', 'w') as outfile:
            outfile.write(accum_pthru)
        with open(destdir+f'/{gn}.othru', 'w') as outfile:
            outfile.write(accum_othru)
        if dataset_name:
            # and add a record to the .textds file
            source_ds = "gata" if "gata" in dataset_name else \
                dataset_name if "extra" in dataset_name else "ftwc"
            with open(make_dsfilepath(destdir, dataset_name), 'a') as dsfile:
                dsfile.write(f'{{"game":"{gn}"')
                dsfile.write(f',"numsteps":{len(playthru)}')
                dsfile.write(f',"source":"{source_ds}"')
                gid, skills = split_gamename(gn)
                if skills:   # if successfully parsed the game name
                    dsfile.write(f',"gid":"{gid}"')
                    quoted_skills = [f'"{s}"' for s in skills]
                    dsfile.write(f',"skills":[{",".join(quoted_skills)}]')
                    #dsfile.write(f',"skills":{skills}') #similar to prev 2 lines, but outputs single, not double quotes
                rtg_json_str = format_rtg_for_json(playthru, rtg=True)
                if rtg_json_str:
                    dsfile.write(',"rtg":' + rtg_json_str)
                taskstackjson = format_taskstacks_for_json(playthru)
                if map_names2ids:
                    taskstackjson = subst_names(taskstackjson, map_names2ids)
                if True:
                    dsfile.write(',"taskstack":' + taskstackjson)
                solverinfo_json = format_solver_info_json(playthru)
                dsfile.write(',"solver":' + solverinfo_json)

                lines = []
                for line in accum_pthru.split('\n'):
                    line = line.strip()
                    if line:
                        lines.append(line)
                dsfile.write(f',"text":{json.dumps(JSON_LINE_SEP.join(lines))}')
                lines = []
                for line in accum_othru.split('\n'):
                    line = line.strip()
                    if line:
                        lines.append(line)
                dsfile.write(f',"text0":{json.dumps(JSON_LINE_SEP.join(lines))}')
                # lines = []
                # raw_accum = "\n".join(raw_all)
                # for line in raw_accum.split('\n'):
                #     line = line.strip()
                #     lines.append(line)
                # dsfile.write(f',"raw":"{JSON_CRLF.join(lines)}"')
                dsfile.write('}\n')
    num_files += 1
    return num_files


def retrieve_playthrough_json(gamename, ptdir=None, gindex: GamesIndex = None, ptid=None):
    # if not ptid:  #allow for multiple playthroughs per game
    #     ptid=playthrough_id(),  # default ptid based on default goal and randseed, can optionally specify

    if gindex:
        ptdir = gindex.get_dir_for_pthru(gamename)
    if ptdir is None:
        ptdir = "./"
    ptdir = normalize_path(ptdir)
    _ptjson = os.path.join(ptdir, gamename+"_PT.json")
    with open(_ptjson, "r") as infile:
        step_array = json.load(infile)
    return step_array


def retrieve_playthrough_nsteps(gamename, ptdir=None, gindex: GamesIndex = None, ptid=None):
    pt_json = retrieve_playthrough_json(gamename, ptdir=ptdir, gindex=gindex, ptid=ptid)
    return len(pt_json)


def find_gn_in_nsteps_index(nsteps_index, gn):
    for i, nsteps_set in enumerate(nsteps_index):
        if gn in nsteps_set:
            return i
    return -1


def get_nsteps_index(ptpatterns:List[str]=None, ptid=None):
    if not ptpatterns:
        ptpatterns = [normalize_path(os.getenv('FTWC_PTHRU', DEFAULT_FTWC_PTHRU_BASE))+"**/*_PT.json"]
    # returns an array -- nsteps_map[n] is a set of gamenames with len(pthru) == n
    nsteps_index = [set()]  # by default, an empty set of games with zero nsteps
    _max_nsteps = 0
    for globpattern in ptpatterns:
        print(globpattern)
        ptfiles = list(glob.glob(globpattern))
        for ptfile in ptfiles:
            #print(ptfile)
            with open(ptfile, "r") as infile:
                assert Path(ptfile).stem.endswith('_PT')
                gn = Path(ptfile).stem[:-3]
                step_array = json.load(infile)
                _nsteps = len(step_array)
                already_indexed = find_gn_in_nsteps_index(nsteps_index, gn)
                if already_indexed > -1:
                    print(f"ALREADY INDEXED ({_nsteps} {already_indexed}) gn={gn}")
                    if already_indexed != _nsteps:
                        assert False, f"!!! ALREADY INDEXED ({_nsteps} {already_indexed}) gn={gn}"

                while _nsteps > _max_nsteps:
                    _max_nsteps = len(nsteps_index)   # _max_nsteps += 1
                    nsteps_index.append(set())    # emtpy set
                nsteps_index[_nsteps].add(gn)
    return nsteps_index


def map_gata_difficulty(game_dirs):
    difficulty_map = defaultdict(set)
    print(f"create_difficulty_map -- game_dirs={game_dirs})")
    for game_dir in game_dirs:
        print("create_difficulty_map -- adding GAME DIR:", game_dir)
        game_names_ = []
        for level in range(1, 11):
            difficulty = f"difficulty_level_{level}"
            print("\n-------------------", difficulty)
            games_list = _list_game_files(game_dir + difficulty)
            game_names_.extend(games_list)
            difficulty_map[level].update(games_list)
        print(f"total games in {game_dir}: {len(game_names_)} {len(set(game_names_))}")
        assert len(game_names_) == len(set(game_names_))  # they should all be unique
    return difficulty_map


def lookup_difficulty_level(gamename, difficulty_map):
    # assert gamename , "Missing required argument: gamename"
    level = -1
    for i in range(1, 11):
        if gamename in difficulty_map[i]:
            level = i
            break
    return level

