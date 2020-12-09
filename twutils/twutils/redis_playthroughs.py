import json
from collections import defaultdict, OrderedDict
import os

from typing import List, Dict, Optional, Any


import textworld
import textworld.gym
import gym


import redis
from redisgraph import Node, Edge, Graph, Path  # https://github.com/RedisGraph/redisgraph-py
from rejson import Client, Path                 # https://github.com/RedisJSON/redisjson-py


from twutils.file_helpers import count_iter_items, parse_gameid, split_gamename
from twutils.twlogic import filter_observables
from twutils.playthroughs import *

from ftwc.wrappers import QaitGym
from ftwc.vocab import WordVocab

TW_DATA_BASEDIR = '/home/guntis/work/github/0_magd3/CodaLab/'
Z8_MAP_DIR = TW_DATA_BASEDIR + 'z8_maps'
XTRACT_DIR = TW_DATA_BASEDIR + 'extracted_data'

print(f"REDIS_FTWCv0 = {REDIS_FTWCv0}")
print(f"REDIS_FTWCv2019 = {REDIS_FTWCv2019}")
print(f"REDIS_EXTRACTED_DATA = {REDIS_EXTRACTED_DATA}")
print(f"REDIS_FTWCv0_TRAINING_GAMES = {REDIS_FTWCv0_TRAINING_GAMES}")
print(f"REDIS_FTWCv2019_TRAINING_GAMES = {REDIS_FTWCv2019_TRAINING_GAMES}")
print(f"REDIS_FTWC_TRAINING = {REDIS_FTWC_TRAINING}")
print("\t\t:[", REDIS_FTWCv0_TRAINING_GAMES, REDIS_FTWCv2019_TRAINING_GAMES, "]")
print(f"REDIS_FTWC_VALID = {REDIS_FTWC_VALID}")
print(f"REDIS_FTWC_TEST = {REDIS_FTWC_TEST}")
print(f"REDIS_FTWC_PLAYTHROUGHS = {REDIS_FTWC_PLAYTHROUGHS}")
print()
print(f"RANDOM SEED for playthroughs: {DEFAULT_PTHRU_SEED}")
print(f"\t\tDefault playthrough ID = :{playthrough_id()}")


training_files = os.listdir(TW_TRAINING_DIR)
# directory contains 3 files per game: *.json, *.ulx, *.z8 
print("Total training files = ", count_iter_items(training_files))
suffixes = ['.json', '.z8', '.ulx']
for suffix in suffixes:
    training_games = list(filter(lambda fname: fname.endswith(suffix), training_files))
    print("number of {} files in {} = {}".format(suffix, TW_TRAINING_DIR, len(training_games)))
game_names_ = [s.split('.')[0] for s in training_games]


def reformat_go_skill( sk ):
    if sk.startswith("go"):
        n = int(sk[2:])
        return f"go{n:2d}"
    else:
        return sk

def split_gamename(gname):
    skills, gid = gname.split('-')
    sklist = skills.split('+')
    return gid, [ reformat_go_skill(sk) for sk in sklist ]
 

def parse_gameid(game_name: str) -> str:
    
    game_id = game_name[11:] if game_name.startswith("tw-cooking-") else game_name
        
    segments = game_id.split('-')
    if len(segments) >= 2:
        code, guid = segments[0:2]
        guid = guid.split('.')[0]
        guid = "{}..{}".format(guid[0:4],guid[-4:])
        segments = code.split('+')
        r, t, g, k, c, o, d = ('0', '0', 0, '*', '*', '*', '*')
        for seg in segments:
            if seg.startswith('recipe'):
                r = seg[len('recipe'):]
            elif seg.startswith('go'):
                g = int(seg[len('go'):])
            elif seg.startswith('take'):
                t = seg[len('take'):]
            elif seg == 'cook':
                k = 'k'
            elif seg == 'cut':
                c = 'c'
            elif seg == 'open':
                o = 'o'
            elif seg == 'drop':
                d = 'd'
            else:
                assert False, "unparsable game_id: {}".format(game_id)
        shortcode = "r{}t{}{}{}{}{}g{:02d}-{}".format(r,t,k,c,o,d,g,guid)
    else:
        shortcode = game_id
    return shortcode


# In[9]:


# print("Num training tw-cooking = ",
#       count_iter_items(filter(lambda gn: gn.startswith('tw-cooking-'), game_names_)))

# for gn in game_names_:
#     assert gn.split('-')[2:] == gn[11:].split('-'), str()

game_names = list(map(lambda gn: gn[11:], game_names_))
# print(len(game_names), game_names[0:2])

games = defaultdict(list)
for gn in game_names:
    skills, gid = gn.split('-')
    if not gid:
        print("SPLIT FAILURE:", gn)
    else:
        games[gid].append(gn) #defaultdict initializes with empty list


# In[10]:


print("Number of training games", len(training_games))
print("Number of unique IDs", len(games))
print()
total = 0
for n in [1,2,3,4,5]:
    c = count_iter_items(filter(lambda g: len(g)==n, games.values()))
    print("IDs with {} entries: {}".format(n, c))
    total += n*c
print("Total =", total)
assert total == len(game_names)


# In[11]:

for list3 in filter(lambda g: len(g)==3, games.values()):  #game seeds with 3 games
    for gn in list3:
        print(split_gamename(gn))




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


# POSTPROCESS extracted .room_state and .desc files to make sure that
# entities are listed in the same order in the room_state list as 
# they are mentioned in the room description.

def check_sort(obj_indexes, exit_indexes, bn, desc, state):
    max_i = -1
    out_of_order = []
    for i,line in obj_indexes:
        if i < max_i:
            out_of_order.append(max_i)
        else:
            max_i = i
    max_i = -1
    for i,line in exit_indexes:
        if i < max_i:
            out_of_order.append(max_i)
        else:
            max_i = i
    if out_of_order: 
        print("============================================")
        print(bn)
        print("-----------------------------")
        print("Out of order:", out_of_order, obj_indexes, exit_indexes)
        print(desc)
        print("-----------------------------")
        print(state)
        return False
    return True

def check_extracted_data(game_names=None, xtract_dir=XTRACT_DIR):
    if not game_names:
        game_names = []
    itercount = 0
    newdir_count = 0
    room_count = 0
    desc_count = 0
    failed_count = 0
    max_desc_len = 0
    max_desc_wlen = 0
    max_desc_str = ''
    for bn in game_names: #[0:10]:
        xdir = xtract_dir + '/' + bn
        itercount += 1
        data_files = os.listdir(xdir)
        # directory contains 3 files per game: *.json, *.ulx, *.z8
    #     print("Total training files = ", count_iter_items(data_files))
        suffixes = ['.room_state', '.desc']
    #     for suffix in suffixes:
        suffix = '.room_state'
        filtered_files = list(filter(lambda fname: fname.endswith(suffix), data_files))
        for f in filtered_files:
            obj_lines = []
            obj_indexes = []
            exit_lines = []
            exit_indexes = []
            room_count += 1
            with open(xdir+'/'+f, 'r') as room_state_file:
                state_str = room_state_file.read()
            state_lines = [line.strip() for line in open(xdir+'/'+f, 'r')]
            room_name = f.split('.')[0]
            with open(xdir+'/'+room_name + '.desc', 'r') as desc_file:
                desc_lines = [line.strip() for line in desc_file]
                desc_str = ' '.join(desc_lines)
                if len(desc_str):
                    desc_count += 1
                    if len(desc_str) > max_desc_len:
                        max_desc_len = len(desc_str)
                        if len(desc_str.split(' ')) > max_desc_wlen:
                            max_desc_wlen = len(desc_str.split(' '))
                            max_desc_str = desc_str
            for line in state_lines:
                entity_name = None
                idx = -1
                if line:
                    if not (line.startswith('--')                              or line.startswith('*')                              or line.startswith('+')                              or line.startswith('~')                              or line.startswith('_')):
                        entity_name = remove_parenthetical_state(line)
                        idx = desc_str.find(entity_name)
                    elif line.startswith('+ '):
                        entity_name = remove_parenthetical_state(line[2:])
                        idx = desc_str.find(entity_name)
                    elif line.startswith('~ '):
    #                 exit_lines.append(remove_parenthetical_state(line[2:]))
                        entity_name = remove_parenthetical_state(line[2:])
                        split = entity_name.split(': ')
                        if split[1] != 'exit':
                            idx = desc_str.find(split[1])
                        else:
                            idx = desc_str.find(' '+split[0])
                    if entity_name:
                        obj_indexes.append((idx, entity_name))
                        if idx < 0:
                            print('****WARNING**** obj:', line, "NOT FOUND IN DESC:", desc_str)
    #         print("Exits:", exit_lines)
    #         print("Objs:", obj_lines)
    #         for line in obj_lines:
    #             idx = desc_str.find(line)
    #             obj_indexes.append((idx, line))
    #         for line in exit_lines:
    #             split = line.split(': ')
    #             if split[1] != 'exit':
    #                 idx = desc_str.find(split[1])
    #             else:
    #                 idx = desc_str.find(' '+split[0])
    #             if idx < 0:
    #                 print('****WARNING**** exit:', line, "NOT FOUND IN DESC:", desc_str)
    #             exit_indexes.append((idx,line))
    #         print(obj_indexes)
            if not check_sort(obj_indexes, exit_indexes, bn, desc_str, state_str):
                failed_count += 1

    print("Processed", itercount, "games, ", room_count, "rooms.", "OUT OF ORDER:", failed_count)
    print("max room description length =", max_desc_len, "# words=", max_desc_wlen, '\n', max_desc_str)
    assert desc_count == room_count

# check_extracted_data(game_names=game_names_)


def add_gamenames_to_redis(rediskey, listofnames):
    if not redserv.exists(rediskey):
        print("Added", redserv.sadd(rediskey, *listofnames), f"FTWC {rediskey} games to Redis")
    else:
        print("Redis has", redserv.scard(rediskey), f"{rediskey} game names")
#
# add_gamenames_to_redis(REDIS_FTWC_TRAINING, game_names_)

# # directory contains 3 files per game: *.json, *.ulx, *.z8
# suffixes = ['.json', '.ulx', '.z8']
# for _dir, rediskey in [ (TW_VALIDATION_DIR, REDIS_FTWC_VALID), (TW_TEST_DIR, REDIS_FTWC_TEST) ]:
#     print()
#     _file_list =  os.listdir(_dir)
#     for suffix in suffixes:
#         _games = list(filter(lambda fname: fname.endswith(suffix), _file_list))
#         print("number of {} files in {} = {}".format(suffix, _dir, len(_games)))
#     _game_names_ = [s.split('.')[0] for s in _games]
#     add_gamenames_to_redis(rediskey, _game_names_)
    


def extracted_data_to_redis(game_names=None, xtract_dir=XTRACT_DIR, redis=None):

    do_write = False

    if not game_names:
        game_names = []
    itercount = 0
    room_count = 0
    desc_count = 0
    failed_count = 0
    max_desc_wlen = 0
    max_desc_wlen = 0
    max_desc_str = ''
    redis_ops = 0
    for bn in game_names: #[0:10]:
        if do_write:
            if not redis.exists(f'{REDIS_EXTRACTED_DATA}:{bn}'):
                redis_ops += 1
                redis.jsonset(f'{REDIS_EXTRACTED_DATA}:{bn}', Path.rootPath(), {'room': {}})
        xdir = xtract_dir + '/' + bn
        itercount += 1
        data_files = os.listdir(xdir)
        # suffixes = ['.room_state', '.desc']
        suffix = '.room_state'
        filtered_files = list(filter(lambda fname: fname.endswith(suffix), data_files))
        sorted_files = list(map( lambda fname: fname.split('_', maxsplit=2), filtered_files ))
        for i, tup in enumerate(sorted_files):
            tup[1] = int(tup[1])  # convert room numbers from ascii to int
            tup[2] = tup[2].split('.')[0]  # the room name
            tup.append(filtered_files[i]) # add the original file name  (NOTE: tup is a list, not a tuple)
        # sort by room number, not alphabetically
        sorted_files = sorted(sorted_files)
#         print(sorted_files[0:3])
        for (_, i, room, f) in sorted_files:
            obj_lines = []
            room_count += 1
            with open(xdir+'/'+f, 'r') as room_state_file:
                state_str = room_state_file.read()
            room_name = f.split('.')[0]
            with open(xdir+'/'+room_name + '.desc', 'r') as desc_file:
                desc_lines = [line.strip() for line in desc_file]
                desc_str = ' '.join(desc_lines)
                if len(desc_str):
                    desc_count += 1
                    if len(desc_str) > len(max_desc_str):
                        if len(desc_str.split(' ')) > max_desc_wlen:
                            max_desc_wlen = len(desc_str.split(' '))
                            max_desc_str = desc_str
            jsonobj = {
                'room_name': room,
                'room_id': i,
                'init_descr': desc_str,
                'init_state': state_str
            }
            if do_write:
                redis_ops += 1
                redis.jsonset(f'{REDIS_EXTRACTED_DATA}:{bn}', Path(f'.room.{room}'), jsonobj)
                #print(bn, room.upper(), f)
            
    print("Processed", itercount, "games, ", room_count, "rooms." )
    print(redis_ops, "redis write ops")
    print("max room description length =", len(max_desc_str), "# words=", max_desc_wlen)
    assert desc_count == room_count

# extracted_data_to_redis(game_names=game_names_, redis=rj)
## Processed 4440 games,  29400 rooms.
## 0 redis write ops
## max room description length = 1266 # words= 245


FTWC_ALL_VOCAB = '/ssd2tb/ftwc/all-vocab.txt'

QAIT_VOCAB = '/ssd2tb/qait/qait_word_vocab.txt'

# FTWC_QAIT_VOCAB = '/ssd2tb/ftwc/combined-qait-ftwc-vocab.txt'
# NOTE: there is only one word in all-vocab.txt not in QAIT_VOCAB:  "bbq's"
# (and it would most likely be split during tokenization)



MAX_PLAYTHROUGH_STEPS = 100
def start_game_for_playthrough(gamefile):
    _word_vocab = WordVocab(vocab_file=QAIT_VOCAB)
    _qgym_ = QaitGym(random_seed=DEFAULT_PTHRU_SEED)
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
                                   max_episode_steps=MAX_PLAYTHROUGH_STEPS-1)
    obs, infos = _qgym_env.reset()
    _word_vocab.init_from_infos_lists(infos['verbs'], infos['entities'])
    return _qgym_env, obs, infos


def step_game_for_playthrough(gymenv, step_cmds:List[str]):
    obs,rewards,dones,infos = gymenv.step(step_cmds)
    return obs, rewards, dones, infos


def save_playthrough_step_info_to_redis(gamename, step_num, obs, rewards, dones, infos, cmds,
                              redisbasekey=REDIS_FTWC_PLAYTHROUGHS, ptid=f'eatmeal_42',
                              do_write=False, redis=None, redis_ops=0):
    print(f"{'++' if do_write else '..'} step:[{step_num}] save_playthrough_step_info: {gamename} ({ptid})")
    world_facts = infos['facts'][0]
    observable_facts, player_room = filter_observables(world_facts, game=infos['game'][0])
    world_facts_serialized = [ f.serialize() for f in world_facts ]
    observable_facts_serialized = [ f.serialize() for f in observable_facts ]
    step_json = {
        f'step_{step_num:02d}': {
            'reward': rewards[0],
            'score': infos['game_score'][0],
            'done': dones[0],
            'player_location': player_room.name,
            'obs': obs[0],
            'feedback': infos['feedback'][0],
            'description': infos['description'][0],
            'inventory': infos['inventory'][0],
            'prev_action': cmds[0],
            'next_action': infos['tw_o_step'][0],
            'possible_actions': infos['admissible_commands'][0],
            'oservable_facts': observable_facts_serialized,
            #'GT_FACTS': world_facts_serialized,
        }
    }
    if step_num == 0:  # at start of game
        step_json['step_00']['GT_FACTS'] = world_facts_serialized
    elif dones[0]:     # finished game
        step_json[f'step_{step_num:02d}']['GT_FACTS'] = world_facts_serialized

    if do_write:
        step_key = list(step_json.keys())[0]  # get the first (and only) key
        redis.jsonset(f'{redisbasekey}:{gamename}',
                      Path(f".{ptid}.{step_key}"), step_json[step_key])
        redis_ops += 1

#    print(f"[{step_num}] **** REWARD: {rewards}  SCORE: {infos['game_score']} DONE: {done} ***********************\n")
#     print("--------------------------- Feedback: ---------------------------\n", infos['feedback'][0])
#     print("--------------------------- Description: ------------------------\n", infos['description'][0])
#     print("--------------------------- Inventory: --------------------------\n", infos['inventory'][0])
#     print("--------------------------- Admissible Commands: ----------------\n", infos['admissible_commands'][0])
#     print("--------------------------- Oracle Next Action: -----------------\n", infos['tw_o_step'][0])
#     #print()
#     print("--------------------------- Facts: ------------------------------\n", infos['facts'][0])
    return redis_ops


def save_playthrough_to_redis(gamename, gamedir=None,
                              redisbasekey=REDIS_FTWC_PLAYTHROUGHS, randseed=DEFAULT_PTHRU_SEED,
                              do_write=False, redis=None, skip_existing=True):

    if gamedir is None:
        if redis:
            if redis.sismember(REDIS_FTWC_TRAINING, gamename):
                gamedir = TW_TRAINING_DIR
            elif redis.sismember(REDIS_FTWC_VALID, gamename):
                gamedir = TW_VALIDATION_DIR
            elif redis.sismember(REDIS_FTWC_TEST, gamename):
                gamedir = TW_TEST_DIR
            else:
                assert False, f"unknown directory for gamename={gamename}"
        else:
            gamedir=TW_TRAINING_DIR   # best guess, we'll see later if it's there or not

    _gamefile = f"{gamedir}/{gamename}.z8"
    if not os.path.exists(_gamefile):
        _gamefile = f"{gamedir}/{gamename}.ulx"

    num_steps = 0
    redis_ops = 0

    ptid = playthrough_id(seed=randseed)  # playtrough ID (which of potentially several different) for this gamename

    if not redis:
        do_write = False

    if do_write:
        if not redis.exists(f'{redisbasekey}:{gamename}'):
            redis_ops += 1
            redis.jsonset(f'{redisbasekey}:{gamename}', Path.rootPath(), {ptid: {}})
        elif skip_existing:
            if redis.jsonobjlen(f'{redisbasekey}:{gamename}', Path('.'+ptid)):  # if exists and non-empty
                print(f"SKIPPED EXISTING playthrough {gamename}")
                return num_steps


    _dones = [0]
    _rewards = [0]
    next_cmds = ['start']
    gymenv, _obs, _infos = start_game_for_playthrough(_gamefile)
    redis_ops = save_playthrough_step_info_to_redis(gamename, num_steps, _obs, _rewards, _dones, _infos, next_cmds,
                                    redisbasekey=redisbasekey,
                                    ptid=ptid,
                                    redis=redis,
                                    do_write=do_write, redis_ops=redis_ops)

    next_cmds = _infos['tw_o_step']
    while not _dones[0] and num_steps < MAX_PLAYTHROUGH_STEPS+1:
        num_steps += 1
        _obs, _rewards, _dones, _infos = step_game_for_playthrough(gymenv, next_cmds)
        redis_ops = save_playthrough_step_info_to_redis(gamename, num_steps, _obs, _rewards, _dones, _infos, next_cmds,
                                            redisbasekey=redisbasekey,
                                            ptid=ptid,
                                            redis=redis,
                                            do_write=do_write, redis_ops=redis_ops)
        next_cmds = _infos['tw_o_step']
    gymenv.close()
    print(f"----------------- {gamename} playthrough steps: {num_steps}  Redis writes {redis_ops} ----------------")
    return num_steps, redis_ops


if __name__ == "__main__":
    import argparse
    from tqdm import tqdm

    gamesets = {'train': REDIS_FTWC_TRAINING, 'valid': REDIS_FTWC_VALID, 'test': REDIS_FTWC_TEST,
                'miniset': REDIS_FTWC_TRAINING }

    def main(args):
        rj = Client(host='localhost', port=6379, decode_responses=True)  # redisJSON API
        total_redis_ops = 0
        if not args.which:
            assert False, "Expected which= one of [train, valid, test, miniset]"
            exit(1)
        rediskey = gamesets[args.which]
        num_games = rj.scard(rediskey)
        gamenames = rj.smembers(rediskey)
        if args.which == 'miniset':
            #gamenames = list(gamenames)[0:3]   # just the first 3
            gamenames = ['tw-cooking-recipe3+take3+cut+go6-Z7L8CvEPsO53iKDg']
        for i, gname in enumerate(tqdm(gamenames)):
            print(f"[{i}] BEGIN PLAYTHOUGH: {gname}")
            num_steps, redis_ops = save_playthrough_to_redis(gname, redis=rj, do_write=args.do_write)
            print(f"[{i}] PLAYTHOUGH {gname}: steps:{num_steps} to redis: {redis_ops}")
            total_redis_ops += redis_ops
        print("Total redis writes:", total_redis_ops)
        if args.do_write:
            rj.save()
        rj.close()

    parser = argparse.ArgumentParser(description="Import playthrough data to redis")
    parser.add_argument("which", choices=('train', 'valid', 'test', 'miniset'))
    parser.add_argument("--do_write", action='store_true')
    args = parser.parse_args()
    main(args)


