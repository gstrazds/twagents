import json
from collections import defaultdict, OrderedDict
import os
import os.path

from typing import List, Dict, Optional, Any

import textworld
import textworld.gym
import gym

from twutils.file_helpers import count_iter_items, split_gamename  # , parse_gameid
from twutils.twlogic import filter_observables
from twutils.gym_wrappers import normalize_feedback_vs_obs_description, simplify_feedback, INSTRUCTIONS_TOKEN

from ftwc.wrappers import QaitGym
from ftwc.vocab import WordVocab

# default directory paths, usually overriden by env, config or cmd line args
TW_GAMES_BASEDIR = '/ssd2tb/ftwc/games/'
TW_TRAINING_DIR = TW_GAMES_BASEDIR + 'train/'
# TW_VALIDATION_DIR = TW_GAMES_BASEDIR + 'valid/'
# TW_TEST_DIR = TW_GAMES_BASEDIR + 'test/'

CMD_START_TOKEN = '>>>['
CMD_END_TOKEN = ']<<<'
GAME_START_CMD = 'start'

DEFAULT_PTHRU_SEED = 42
GOAL_MEAL = 'eatmeal'
GOAL_ANSWER_WHERE = 'whereis'
GOAL_ANSWER_EXISTS = 'exists'


def normalize_path(path_str: str, subdir: str = None):
    if subdir:
        path_str = os.path.join(path_str, subdir)
    path_str = os.path.normpath(path_str)
    path_str = os.path.join(path_str, '')   # make sure it ends with a trailing slash
    return path_str


def get_games_dir(basepath: str = None, splitname: str = 'train'):
    if not basepath:
        basepath = os.getenv('TW_GAMES_BASEDIR', TW_GAMES_BASEDIR)
    games_dir = basepath
    return normalize_path(games_dir, splitname)


def playthrough_id(objective_name=GOAL_MEAL, seed=None):
    if seed is None:
        seed = DEFAULT_PTHRU_SEED
    return f"{objective_name}_{seed}"


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


FTWC_ALL_VOCAB = '/ssd2tb/ftwc/all-vocab.txt'

QAIT_VOCAB = '/ssd2tb/qait/qait_word_vocab.txt'

# FTWC_QAIT_VOCAB = '/ssd2tb/ftwc/combined-qait-ftwc-vocab.txt'
# NOTE: there is only one word in all-vocab.txt not in QAIT_VOCAB:  "bbq's"
# (and it would most likely be split during tokenization)


MAX_PLAYTHROUGH_STEPS = 150
def start_game_for_playthrough(gamefile,
                               raw_obs_feedback=True,  # don't apply ConsistentFeedbackWrapper
                               passive_oracle_mode=False,  # if True, don't predict next action
                               max_episode_steps=MAX_PLAYTHROUGH_STEPS,
                               random_seed=DEFAULT_PTHRU_SEED
                               ):  #
    _word_vocab = WordVocab(vocab_file=QAIT_VOCAB)
    _qgym_ = QaitGym(random_seed=random_seed,
                     raw_obs_feedback=raw_obs_feedback,
                     passive_oracle_mode=passive_oracle_mode)
    _qgym_env = _qgym_.make_batch_env([gamefile],
                                   _word_vocab,  # vocab not really needed by Oracle, just for gym.space
                                   request_infos=textworld.EnvInfos(
                                        feedback=True,
                                        description=True,
                                        inventory=True,
                                        location=True,
                                        entities=True,
                                        verbs=True,
                                        facts=True,   # use ground truth facts about the world (since this is a training oracle)
                                        admissible_commands=True,
                                        game=True,
                                        extras=["recipe", "uuid"]
                                   ),
                                   batch_size=1,
                                   max_episode_steps=max_episode_steps)
    obs, infos = _qgym_env.reset()
    _word_vocab.init_from_infos_lists(infos['verbs'], infos['entities'])
    return _qgym_env, obs, infos


def step_game_for_playthrough(gymenv, step_cmds:List[str]):
    obs,rewards,dones,infos = gymenv.step(step_cmds)
    return obs, rewards, dones, infos


def format_stepkey(step_num:int):
    return f'step_{step_num:02d}'


def playthrough_step_to_json(cmds, dones, infos, obs, rewards, step_num):
    world_facts = infos['facts'][0]
    observable_facts, player_room = filter_observables(world_facts, game=infos['game'][0])
    world_facts_serialized = [f.serialize() for f in world_facts]
    observable_facts_serialized = [f.serialize() for f in observable_facts]
    step_key = format_stepkey(step_num)
    oracle_action = infos['tw_o_step'][0] if 'tw_o_step' in infos else None
    oracle_state = infos['tw_o_state'][0] if 'tw_o_state' in infos else None
    step_json = {
        step_key: {
            'reward': rewards[0],
            'score': infos['game_score'][0],
            'done': dones[0],
            'player_location': player_room.name,
            'obs': obs[0],
            'feedback': infos['feedback'][0],
            'description': infos['description'][0],
            'inventory': infos['inventory'][0],
            'prev_action': cmds[0],
            'next_action': oracle_action,
            'possible_actions': infos['admissible_commands'][0],
            'obs_facts': observable_facts_serialized,
            # 'GT_FACTS': world_facts_serialized,
        }
    }
    if oracle_state:
        step_json[step_key]['tw_o_state'] = oracle_state
    if step_num == 0 or dones[0]:  # at start of game or game over
        step_json[step_key]['GT_FACTS'] = world_facts_serialized
    return step_json


def playthrough(gamefile, randseed=DEFAULT_PTHRU_SEED):
    num_steps = 0
    step_array = []  # convert json dict data (with redundant keys) to an array for convenience

    _dones = [0]
    _rewards = [0]
    next_cmds = ['start']
    gymenv, _obs, _infos = start_game_for_playthrough(gamefile, random_seed=randseed)
    playthru_step_data = playthrough_step_to_json(next_cmds, _dones, _infos, _obs, _rewards, num_steps)

    step_array.append(playthru_step_data[format_stepkey(num_steps)])

    next_cmds = _infos['tw_o_step']
    while not _dones[0] and num_steps < MAX_PLAYTHROUGH_STEPS+1:
        num_steps += 1
        _obs, _rewards, _dones, _infos = step_game_for_playthrough(gymenv, next_cmds)
        playthru_step_data = playthrough_step_to_json(next_cmds, _dones, _infos, _obs, _rewards, num_steps)
        step_array.append(playthru_step_data[format_stepkey(num_steps)])
        next_cmds = _infos['tw_o_step']
    gymenv.close()
    return step_array


from textworld.logic import Proposition  #, Variable, Signature, State
from symbolic.knowledge_graph import KnowledgeGraph


def format_facts(facts_list, prev_action=None, obs_descr=None, kg=None):
    if not isinstance(facts_list[0], Proposition):   # maybe serialized list of facts
        # world_facts0 = world_facts.copy()
        facts_list = [Proposition.deserialize(fact_json) for fact_json in facts_list]

    if kg is None:
        kg = KnowledgeGraph(None, debug=False)   # suppress excessive print() outs
    kg.update_facts(facts_list, prev_action=prev_action)
    #return str(kg)
    return kg.describe_room(kg.player_location.name, obs_descr=obs_descr)


def get_kg_descr(kg_accum, stepdata):
    obs_facts_key = 'obs_facts'
    # obs_facts_key = 'oservable_facts' if 'oservable_facts' in stepdata else 'obs_facts'  # TODO: delete this line
    prev_action = stepdata['prev_action']
    observed_facts = stepdata[obs_facts_key]
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
    kg_descr = format_facts(observed_facts, kg=kg_accum, prev_action=prev_action, obs_descr=stepdata['description'])
    return kg_descr


def format_playthrough_step(kg_descr, stepdata, simplify_raw_obs_feedback=True):
    feedback = stepdata['feedback']
    prev_action = stepdata['prev_action']
    if simplify_raw_obs_feedback:
        new_feedback = normalize_feedback_vs_obs_description(prev_action,
                                                             stepdata['obs'],
                                                             stepdata['feedback'],
                                                             stepdata['description'])
        if new_feedback:
            # print(f"export_playthru MODIFYING ['feedback'] : '{new_feedback}' <-- orig:", stepdata['feedback'])
            feedback = new_feedback

   # print(f"[{i}] {gn} .....")
    outstr = f"\n{CMD_START_TOKEN} {prev_action} {CMD_END_TOKEN}\n"
    pthru = outstr
    # pthru_out += outstr
    if feedback: #and prev_action != 'start':
        outstr += feedback
        # feedback = feedback.strip()
    pthru += simplify_feedback(feedback) + '\n'
    pthru += kg_descr + '\n'
    # outstr += '\n' + stepdata['obs']
    outstr += '\n' + stepdata['description']
    outstr += '\n\n' + stepdata['inventory'] + '\n'
    return outstr, pthru


def concat_pthru_step(pthru_so_far:str, pthru_step:str, keep_objectives=True) -> str:
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


class GamesIndex:
    def __init__(self):
        self._index = {}   # maps game_name => {'skills': list of skills, 'dir': index in _game_dirs}
        self.game_dirs = []
        self.pthru_dirs = []

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
            idx_ptdir = self.pthru_dirs.index(pthrus_dir)
        except ValueError:
            self.pthru_dirs.append(pthrus_dir)
            dir_index = len(self.pthru_dirs)-1

        n_dupl = 0
        n_added = 0
        n_updated = 0
        for gn in game_names:
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
        print(f"Added {n_added} pthtrus + {n_dupl}  updated  + {n_dupl} unchanged")

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
