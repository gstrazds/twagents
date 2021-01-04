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
from twutils.gym_wrappers import normalize_feedback_vs_obs_description
from twutils.redis_ids import *
from twutils.playthroughs import *


# default values, can be overriden by config or cmd line args
TW_DATA_BASEDIR = '/home/guntis/work/github/0_magd3/CodaLab/'
Z8_MAP_DIR = TW_DATA_BASEDIR + 'z8_maps'
XTRACT_DIR = TW_DATA_BASEDIR + 'extracted_data'

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

FTWC_GAME_SETS = (REDIS_FTWC_TRAINING, REDIS_FTWC_VALID, REDIS_FTWC_TEST)
GATA_GAME_SETS = (REDIS_GATA_TRAINING, REDIS_GATA_VALID, REDIS_GATA_TEST)


def create_skills_map(redserv=None, skillsmap_key=REDIS_FTWC_SKILLS_MAP, gameset_keys=FTWC_GAME_SETS):
    """ after all game names have been added to redis, we map skills to game names"""
    if redserv is None:
        _rj = Client(host='localhost', port=6379, decode_responses=True)
    else:
        _rj = redserv
    skills_index = {}  # maps skillname to a set of game names
    all_mapped_skills = set()  # all game names that are in the skills map
    for setkey in gameset_keys:
        game_names = _rj.smembers(setkey)
        print(f"{setkey} has {len(game_names)} members")

        for g in game_names:
            gid, sklist = split_gamename(g)
            # print(g, gid, sklist)
            for skill in sklist:
                if skill not in skills_index:
                    skills_index[skill] = set()

                skills_index[skill].add(g)
                _rj.sadd(skillsmap_key+skill, g)

    # print(len(skills_index), skills_index.keys())
    # for key in skills_index.keys():
    #     print(key, len(skills_index[key]))
    #     all_mapped_skills = all_mapped_skills.union(skills_index[key])


    skillsmap_keys = _rj.keys(skillsmap_key + "*")

    for k in skillsmap_keys:
        print(k, _rj.scard(k))
        all_mapped_skills = all_mapped_skills.union(_rj.smembers(k))

    print(f"TOTAL # of game files for which skills have been mapped: {len(all_mapped_skills)}")
    if redserv is None:
        _rj.close()

# create_skills_map(skillsmap_key=REDIS_FTWC_SKILLS_MAP, gameset_keys=FTWC_GAME_SETS)

# create_skills_map(skillsmap_key=REDIS_GATA_SKILLS_MAP, gameset_keys=GATA_GAME_SETS)
# gata:v1:training-games has 1000 members
# gata:v1:valid-games has 200 members
# gata:v1:test-games has 200 members
# gata:v1:skills:go09 140
# gata:v1:skills:cook 560
# gata:v1:skills:recipe3 420
# gata:v1:skills:take3 420
# gata:v1:skills:take1 980
# gata:v1:skills:go12 280
# gata:v1:skills:go06 420
# gata:v1:skills:cut 560
# gata:v1:skills:open 1400
# gata:v1:skills:recipe1 980
# TOTAL # of game files for which skills have been mapped: 1400

def count_game_files(dirpath):
    game_names_ = []
    all_files = os.listdir(dirpath)
    # directory contains up to 3 files per game: *.json, *.ulx, *.z8
    print(f"Total files in {dirpath} = {count_iter_items(all_files)}" )
    suffixes = ['.ulx', '.z8', '.json']
    for suffix in suffixes:
        game_files = list(filter(lambda fname: fname.endswith(suffix), all_files))
        if game_files:
            print("number of {} files in {} = {}".format(suffix, dirpath, len(game_files)))
            game_names_ = [s.split('.')[0] for s in game_files]
    return game_names_

def create_difficulty_map(redserv=None,
                          difficulty_map_key=REDIS_GATA_DIFFICULTY_MAP,
                          game_dirs=None):
    if redserv is None:
        _rj = Client(host='localhost', port=6379, decode_responses=True)
    else:
        _rj = redserv

    if game_dirs is None:
        training_dir = _rj.hget(REDIS_DIR_MAP, 'GATA_TRAINING_DIR')
        validation_dir = _rj.hget(REDIS_DIR_MAP, 'GATA_VALIDATION_DIR')
        test_dir = _rj.hget(REDIS_DIR_MAP, 'GATA_TEST_DIR')
        game_dirs = (training_dir, validation_dir, test_dir)
    print(f"create_difficulty_map({difficulty_map_key}, game_dirs={game_dirs})")
    for game_dir in game_dirs:
        print("create_difficulty_map -- GAME DIR:", game_dir)
        game_names_ = []
        for level in range(1, 11):
            difficulty = f"difficulty_level_{level}"
            print("\n-------------------", difficulty)
            games_list = count_game_files(game_dir + difficulty)
            game_names_.extend(games_list)
            _rj.sadd(difficulty_map_key+str(level), *games_list)
        print(f"total games in {game_dir}: {len(game_names_)} {len(set(game_names_))}")
        assert len(game_names_) == len(set(game_names_))  # they should all be unique
    if redserv is None:
        _rj.close()

def create_nsteps_map(redserv=None,
                           nsteps_map_key=REDIS_FTWC_NSTEPS_MAP,
                           nsteps_index_key=REDIS_FTWC_NSTEPS_INDEX,
                           gameset_keys=FTWC_GAME_SETS):
    """ after all playthroughs have been save to redis, index number of steps <-> game names """
    if redserv is None:
        _rj = Client(host='localhost', port=6379, decode_responses=True)
    else:
        _rj = redserv

    if nsteps_map_key == REDIS_GATA_NSTEPS_MAP:
        redisbasekey = REDIS_GATA_PLAYTHROUGHS
    elif nsteps_map_key == REDIS_FTWC_NSTEPS_MAP:
        redisbasekey = REDIS_FTWC_PLAYTHROUGHS
    else:
        assert False, "Unknown Redis nsteps_map_key "+nsteps_map_key

    for key in _rj.keys(nsteps_index_key + "*"):
        print("Will delete:", key)
        _rj.delete(key)
    print(_rj.hlen(nsteps_map_key))
    _rj.delete(nsteps_map_key)

    for setkey in gameset_keys:
        game_names_ = _rj.smembers(setkey)
        for _gn in game_names_:
            nsteps = retrieve_playthrough_nsteps(_gn, redis=_rj, redisbasekey=redisbasekey)
            if nsteps > 0:
                print(nsteps, _gn)
                _rj.hset(nsteps_map_key, _gn, nsteps)
                _rj.sadd(f"{nsteps_index_key}{nsteps}", _gn)

    print(len(_rj.keys(nsteps_index_key+"*")), _rj.hlen(nsteps_map_key))
    total = 0
    sort_list = []
    for key in _rj.keys(nsteps_index_key+"*"):
        nsteps = int(key.split(':')[-1])
        num_games = _rj.scard(key)
        total += num_games
        sort_list.append( (nsteps, num_games, key) )
        # print(key,  "has", num_games, "game names")
    sort_list.sort()
    for nsteps, num_games, setkey in sort_list:
        print(f"[{nsteps}]\t {num_games}\t {setkey}")
    if redserv is None:
        _rj.close()

# NSTEPS MAPS can be created only AFTER playthrough data has been saved to Redis
# create_nsteps_map(nsteps_map_key=REDIS_FTWC_NSTEPS_MAP, nsteps_index_key=REDIS_FTWC_NSTEPS_INDEX, gameset_keys=FTWC_GAME_SETS)
# create_nsteps_map(nsteps_map_key=REDIS_GATA_NSTEPS_MAP, nsteps_index_key=REDIS_GATA_NSTEPS_INDEX, gameset_keys=GATA_GAME_SETS)

def select_games_for_mingpt(redserv=None):
    if redserv is None:
        _rj = Client(host='localhost', port=6379, decode_responses=True)
    else:
        _rj = redserv
    # for n in range(70):
    # rkey = f"{REDIS_FTWC_NSTEPS_INDEX}{n}"
    #     if rj.exists(rkey):
    #         n_nsteps = rj.scard(f"{REDIS_FTWC_NSTEPS_INDEX}{n}")
    #         n_nodrop = rj.sdiff(f"{REDIS_FTWC_NSTEPS_INDEX}{n}", "ftwc:cog2019:skills:drop")
    # #         maxtoks, mintoks = get_max_token_length(n_nodrop)
    #         if n <= 32 and len(n_nodrop):
    #             rj.sadd(REDIS_MINGPT_ALL, *n_nodrop)
    #         print(f"{n:2d}:__{n_nsteps:3d}__{len(n_nodrop):3d}") #" {maxtoks:4d} {mintoks:4d}")

    if redserv is None:
        _rj.close()

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




def save_playthrough_step_info_to_redis(gamename, step_num, obs, rewards, dones, infos, cmds,
                              redisbasekey=REDIS_FTWC_PLAYTHROUGHS, ptid=f'eatmeal_42',
                              do_write=False, redis=None, redis_ops=0):
    print(f"{'++' if do_write else '..'} step:[{step_num}] save_playthrough_step_info: {gamename} ({ptid})")
    step_json = playthrough_step_to_json(cmds, dones, infos, obs, rewards, step_num)

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
    return redis_ops, step_json


def save_playthrough_to_redis(gamename, gamedir=None,
                              redisbasekey=REDIS_FTWC_PLAYTHROUGHS, randseed=DEFAULT_PTHRU_SEED, goal_type=GOAL_MEAL,
                              do_write=False, redis=None, skip_existing=True):

    if gamedir is None:
        if redis:
            gamedir = get_dir_for_game(redis, gamename)
        else:
            gamedir=TW_TRAINING_DIR   # best guess, we'll see later if it's there or not

    _gamefile = f"{gamedir}/{gamename}.z8"
    if not os.path.exists(_gamefile):
        _gamefile = f"{gamedir}/{gamename}.ulx"

    redis_ops = 0
    num_steps = 0

    ptid = playthrough_id(objective_name=goal_type, seed=randseed)  # playtrough ID (which of potentially different) for this gamename

    if not redis:
        do_write = False

    if do_write:
        if not redis.exists(f'{redisbasekey}:{gamename}'):
            redis_ops += 1
            redis.jsonset(f'{redisbasekey}:{gamename}', Path.rootPath(), {ptid: {}})
        elif skip_existing:
            if redis.jsonobjlen(f'{redisbasekey}:{gamename}', Path('.'+ptid)):  # if exists and non-empty
                print(f"SKIPPED EXISTING playthrough {gamename}")
                return num_steps, redis_ops

    _dones = [0]
    _rewards = [0]
    next_cmds = ['start']
    gymenv, _obs, _infos = start_game_for_playthrough(_gamefile)
    redis_ops, _ = save_playthrough_step_info_to_redis(gamename, num_steps, _obs, _rewards, _dones, _infos, next_cmds,
                                    redisbasekey=redisbasekey,
                                    ptid=ptid,
                                    redis=redis,
                                    do_write=do_write, redis_ops=redis_ops)

    next_cmds = _infos['tw_o_step']
    while not _dones[0] and num_steps < MAX_PLAYTHROUGH_STEPS+1:
        num_steps += 1
        _obs, _rewards, _dones, _infos = step_game_for_playthrough(gymenv, next_cmds)
        redis_ops, _ = save_playthrough_step_info_to_redis(gamename, num_steps, _obs, _rewards, _dones, _infos, next_cmds,
                                            redisbasekey=redisbasekey,
                                            ptid=ptid,
                                            redis=redis,
                                            do_write=do_write, redis_ops=redis_ops)
        next_cmds = _infos['tw_o_step']
    gymenv.close()
    print(f"----------------- {gamename} playthrough steps: {num_steps}  Redis writes {redis_ops} ----------------")
    return num_steps, redis_ops


def get_difficulty_mapping(redserv, gamename):
    for difficulty in range(1,11):
        difficulty = str(difficulty)
        if redserv.sismember(REDIS_GATA_DIFFICULTY_MAP+difficulty, gamename):
            return difficulty
    print("!! WARNING: not found in REDIS_GATA_DIFFICULTY_MAP:", gamename)
    return "0"

def get_dir_for_game(redserv, gamename):
    if redserv.sismember(REDIS_FTWC_TRAINING, gamename):
        gamedir = redserv.hget(REDIS_DIR_MAP, 'TW_TRAINING_DIR')
    elif redserv.sismember(REDIS_FTWC_VALID, gamename):
        gamedir = redserv.hget(REDIS_DIR_MAP, 'TW_VALIDATION_DIR')
    elif redserv.sismember(REDIS_FTWC_TEST, gamename):
        gamedir = redserv.hget(REDIS_DIR_MAP, 'TW_TEST_DIR')

    elif redserv.sismember(REDIS_GATA_TRAINING, gamename):
        gamedir = redserv.hget(REDIS_DIR_MAP, 'GATA_TRAINING_DIR')
        difficulty = get_difficulty_mapping(redserv, gamename)
        gamedir += f"difficulty_level_{difficulty}/"

    elif redserv.sismember(REDIS_GATA_VALID, gamename):
        gamedir = redserv.hget(REDIS_DIR_MAP, 'GATA_VALIDATION_DIR')
        difficulty = get_difficulty_mapping(redserv, gamename)
        gamedir += f"difficulty_level_{difficulty}/"

    elif redserv.sismember(REDIS_GATA_TEST, gamename):
        gamedir = redserv.hget(REDIS_DIR_MAP, 'GATA_TEST_DIR')
        difficulty = get_difficulty_mapping(redserv, gamename)
        gamedir += f"difficulty_level_{difficulty}/"

    else:
        print(f"WARNINIG: unknown directory for gamename={gamename}")
        return None
    return gamedir


def get_dir_for_pthru(redserv, gamename):
    if redserv.sismember(REDIS_MINGPT_TRAINING, gamename):
        gamedir = redserv.hget(REDIS_DIR_MAP, 'MINGPT_TRAIN_DIR')
    elif redserv.sismember(REDIS_MINGPT_VALID, gamename):
        gamedir = redserv.hget(REDIS_DIR_MAP, 'MINGPT_VALID_DIR')
    elif redserv.sismember(REDIS_MINGPT_TEST, gamename):
        gamedir = redserv.hget(REDIS_DIR_MAP, 'MINGPT_TEST_DIR')
    elif redserv.sismember(REDIS_FTWC_TRAINING, gamename):
        gamedir = redserv.hget(REDIS_DIR_MAP, 'FTWC_TRAIN_PTHRUS')
    elif redserv.sismember(REDIS_FTWC_VALID, gamename):
        gamedir = redserv.hget(REDIS_DIR_MAP, 'FTWC_VALID_PTHRUS')
    elif redserv.sismember(REDIS_FTWC_TEST, gamename):
        gamedir = redserv.hget(REDIS_DIR_MAP, 'FTWC_TEST_PTHRUS')

    elif redserv.sismember(REDIS_GATA_TRAINING, gamename):
        gamedir = redserv.hget(REDIS_DIR_MAP, 'GATA_TRAIN_PTHRUS')
    elif redserv.sismember(REDIS_GATA_VALID, gamename):
        gamedir = redserv.hget(REDIS_DIR_MAP, 'GATA_VALID_PTHRUS')
    elif redserv.sismember(REDIS_GATA_TEST, gamename):
        gamedir = redserv.hget(REDIS_DIR_MAP, 'GATA_TEST_PTHRUS')
    else:
        print(f"WARNINIG: unknown directory for gamename={gamename}")
        return None
    return gamedir


def retrieve_playthrough_json(
        gamename,
        redis=None,  # redis server: expected to be a RedisJSON client connection
        ptid=playthrough_id(),  # default playthrough, can optionally specify
        redisbasekey=REDIS_FTWC_PLAYTHROUGHS, randseed=DEFAULT_PTHRU_SEED):

    if redis is None:
        _rj = Client(host='localhost', port=6379, decode_responses=True)
    else:
        assert isinstance(redis, Client)
        _rj = redis
    jsonobj = _rj.jsonget(f'{redisbasekey}:{gamename}', Path('.'+ptid))
    if redis is None:
        _rj.close()
    step_array = []  # convert json dict data (with redundant keys) to an array for convenience
    for i, step_key in enumerate(list(jsonobj.keys())):
        assert step_key == format_stepkey(i)   # format_stepkey() from twutils.playthroughs
        step_array.append(jsonobj[step_key])
    assert len(step_array) == len(jsonobj.keys())
    return step_array


def retrieve_playthrough_nsteps(
        gamename,
        redis=None,  # redis server: expected to be a RedisJSON client connection
        ptid=playthrough_id(),  # default playthrough, can optionally specify
        redisbasekey=REDIS_FTWC_PLAYTHROUGHS, randseed=DEFAULT_PTHRU_SEED):

    if redis is None:
        _rj = Client(host='localhost', port=6379, decode_responses=True)
    else:
        assert isinstance(redis, Client)
        _rj = redis
    nsteps = _rj.jsonobjlen(f'{redisbasekey}:{gamename}', Path(f".{ptid}"))
    if redis is None:
        _rj.close()
    return nsteps
    # if stepkeys:
    #     for k in stepkeys:
    #         assert k.startswith("step_")
    #     return len(stepkeys)
    # return 0


def export_playthru(gn, destdir='.', redis=None, redisbasekey=REDIS_FTWC_PLAYTHROUGHS):

    # gn = 'tw-cooking-recipe3+take3+cut+go6-Z7L8CvEPsO53iKDg'
    # gn = 'tw-cooking-recipe1+cook+cut+open+drop+go6-xEKyIJpqua0Gsm0q'
    #_rj = Client(host='localhost', port=6379, decode_responses=True)

    playthru = retrieve_playthrough_json(gn, redis=redis, redisbasekey=redisbasekey)
    # ! NB: the following code modifies the contents of the retrieved playthru json in memory (but not in Redis)
    pthru_all = ''
    kg_accum = KnowledgeGraph(None, debug=False)   # suppress excessive print() outs
    num_files = 0
    for i, stepdata in enumerate(playthru):
        prev_action = stepdata['prev_action']
        if prev_action and " the " in prev_action:
            prev_action = prev_action.replace(" the ", " ")
            stepdata['prev_action'] = prev_action

        kg_descr = get_kg_descr(kg_accum, stepdata)
        prev_options = kg_accum.set_formatting_options('parsed-obs')
        assert prev_options == 'kg-descr', prev_options
        kg_descr_without_oracle = get_kg_descr(kg_accum, stepdata)
        kg_accum.set_formatting_options(prev_options)

        # because saved playthroughs have raw feedback and obs, do what the ConsistentFeedbackWrapper would normally do
        outstr, pthru = format_playthrough_step(kg_descr, stepdata, simplify_raw_obs_feedback=True)
        _, pthru0 = format_playthrough_step(kg_descr_without_oracle, stepdata, simplify_raw_obs_feedback=True)

        pthru_all = concat_pthru_step(pthru_all, pthru, keep_objectives=True)
        xdir = destdir + '/' + gn
        if not os.path.exists(xdir):
            os.makedirs(xdir)
        with open(xdir+f'/_step_{i:02d}.txt', 'w') as outfile:
            outfile.write(outstr)
            num_files += 1
        with open(xdir + f'/step_{i:02d}.pthru', 'w') as outfile:
            outfile.write(pthru)
            num_files += 1
        with open(xdir + f'/step_{i:02d}.othru', 'w') as outfile:
            outfile.write(pthru0)
            num_files += 1
    with open(destdir+f'/{gn}.pthru', 'w') as outfile:
        outfile.write(pthru_all)
        num_files += 1
    return num_files


if __name__ == "__main__":
    import argparse
    from tqdm import tqdm

    gamesets = {'train': REDIS_FTWC_TRAINING, 'valid': REDIS_FTWC_VALID, 'test': REDIS_FTWC_TEST,
                'miniset': REDIS_FTWC_TRAINING,
                'gata_train': REDIS_GATA_TRAINING, 'gata_valid': REDIS_GATA_VALID, 'gata_test': REDIS_GATA_TEST,
                }

    def main(args):
        rj = Client(host='localhost', port=6379, decode_responses=True)  # redisJSON API
        total_redis_ops = 0
        total_files = 0
        if args.export_files:
            print("Exporting playthough data from Redis to files")
        else:
            print("Importing playrhroughs to Redis...")

        if not args.which:
            assert False, "Expected which= one of [train, valid, test, miniset, gata_train, gata_valid, gata_test]"
            exit(1)
        rediskey = gamesets[args.which]
        if (args.which).startswith("gata_"):
            redisbasekey = REDIS_GATA_PLAYTHROUGHS
        else:
            redisbasekey = REDIS_FTWC_PLAYTHROUGHS
        num_games = rj.scard(rediskey)
        gamenames = rj.smembers(rediskey)
        if args.which == 'miniset':
            #gamenames = list(gamenames)[0:3]   # just the first 3
            gamenames = ['tw-cooking-recipe3+take3+cut+go6-Z7L8CvEPsO53iKDg']
        for i, gname in enumerate(tqdm(gamenames)):
            if not args.export_files:
                print(f"[{i}] BEGIN PLAYTHROUGH: {gname}")
                num_steps, redis_ops = save_playthrough_to_redis(gname, redis=rj,
                                                                 redisbasekey=redisbasekey,
                                                                 do_write=args.do_write)
                print(f"[{i}] PLAYTHROUGH {gname}: steps:{num_steps} to redis: {redis_ops}")
                total_redis_ops += redis_ops
            else:
                if (args.which).startswith("gata_"):
                    destdir = f'./playthru_data/{args.which}'
                else:
                    destdir = f'./playthru_data/{args.which}'
                total_files += export_playthru(gname, destdir=destdir, redis=rj, redisbasekey=redisbasekey)
        print("Total redis writes:", total_redis_ops)
        print("Total files exported:", total_files)
        if args.do_write and not args.export_files:
            rj.save()
        rj.close()

    parser = argparse.ArgumentParser(description="Import playthrough data to redis")
    parser.add_argument("which", choices=('train', 'valid', 'test', 'miniset', 'gata_train', 'gata_valid', 'gata_test'))
    parser.add_argument("--export_files", action='store_true')
    parser.add_argument("--do_write", action='store_true')
    args = parser.parse_args()
    main(args)



