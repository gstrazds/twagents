import os
import json
import gym
from typing import List, Dict, Optional, Any
import numpy as np
import torch

import textworld
import textworld.gym
from textworld import EnvInfos
from textworld.generator.game import Game, EntityInfo
import rlpyt.envs.gym

from symbolic.game_agent import TextGameAgent
from symbolic.task_modules import RecipeReaderTask
from symbolic.task_modules.navigation_task import ExploreHereTask
from twutils.twlogic import filter_observables
from symbolic.entity import MEAL
# from symbolic.event import NeedToAcquire, NeedToGoTo, NeedToDo
from .vocab import WordVocab, WordSpace

# ToTensor gym env wrapoer copied from PyTorch-Lightning-Bolts
class ToTensor(gym.Wrapper):
    """Converts env outputs to torch Tensors"""

    def __init__(self, env=None):
        super(ToTensor, self).__init__(env)

    def step(self, action):
        """Take 1 step and cast to tensor"""
        state, reward, done, info = self.env.step(action)
        return torch.tensor(state), torch.tensor(reward), done, info

    def reset(self):
        """reset the env and cast to tensor"""
        return torch.tensor(self.env.reset())

# adapted from QAit (customized) version of textworld.generator.game.py
def game_objects(game: Game) -> List[EntityInfo]:
    """ The entity information of all relevant objects in this game. """
    def _filter_unnamed_and_room_entities(e):
        return e.name and e.type != "r"
    return filter(_filter_unnamed_and_room_entities, game.infos.values())


def game_object_names(game: Game) -> List[str]:
    """ The names of all objects in this game. """
    get_names_method = getattr(game, "object_names", None)  # game from TextWorld-QAit (customized)
    if get_names_method and callable(get_names_method):
        print(f"game_object_names<TextWorld-QAit>({game.metadata['uuid']}")
        return game.get_names_method()
    get_names_method = getattr(game, "objects_names", None)  # game from TextWorld (1.3.x)
    if get_names_method and callable(get_names_method):
        print(f"game_object_names<TextWorld-1.3.x>({game.metadata['uuid']}")
        return game.get_names_method()
    print(f"game_object_names<local -- game_objects()>({game.metadata['uuid']}")
    return [entity.name for entity in game_objects(game)]


def game_object_nouns(game: Game) -> List[str]:
    """ The noun parts of all objects in this game. """
    _object_nouns = [entity.noun for entity in game_objects(game)]
    return _object_nouns


def game_object_adjs(game: Game) -> List[str]:
    """ The adjective parts of all objects in this game. """
    _object_adjs = [entity.adj if entity.adj is not None else '' for entity in game_objects(game)]
    return _object_adjs


# from challenge.py (invoked during game generation)
# Collect infos about this game.
# metadata = {
#     "seeds": options.seeds,
#     "settings": settings,
# }
#
# def _get_fqn(obj):
#     """ Get fully qualified name """
#     obj = obj.parent
#     name = ""
#     while obj:
#         obj_name = obj.name
#         if obj_name is None:
#             obj_name = "inventory"
#         name = obj_name + "." + name
#         obj = obj.parent
#     return name.rstrip(".")
#
# object_locations = {}
# object_attributes = {}
# for name in game.object_names:
#     entity = M.find_by_name(name)
#     if entity:
#         if entity.type != "d":
#             object_locations[name] = _get_fqn(entity)
#         object_attributes[name] = [fact.name for fact in entity.properties]
#
# game.extras["object_locations"] = object_locations
# game.extras["object_attributes"] = object_attributes
#
# game.metadata = metadata
# uuid = "tw-interactive_qa-{specs}-{seeds}"
# uuid = uuid.format(specs=encode_seeds((options.nb_rooms, options.nb_objects)),
#                    seeds=encode_seeds([options.seeds[k] for k in sorted(options.seeds)]))
# game.metadata["uuid"] = uuid
# game.extras["uuid"] = uuid

# UNUSED BY CURRENT CODE
# request_all_infos = EnvInfos(
#                            description=True,
#                            inventory=True,
#                            # location=True,
#                            facts=True,
#                            last_action=True,
#                            admissible_commands=True,
#     # static infos, don't change during the game:
#                            game=True,
#                            verbs=True,
#                            # location_names=True, # !QAIT-SPECIFIC
#                            # location_nouns=True, # !QAIT-SPECIFIC
#                            # location_adjs=True,  # !QAIT-SPECIFIC
#                            object_names=True,   # !QAIT-SPECIFIC
#                            object_nouns=True,   # !QAIT-SPECIFIC
#                            object_adjs=True,    # !QAIT-SPECIFIC
#                            extras=[                      # the extras come from challenge.py
#                                    "object_locations",   # !QAIT-SPECIFIC
#                                    "object_attributes",  # !QAIT-SPECIFIC
#                                    "uuid"]              # !QAIT-SPECIFIC
#                            )

request_qait_info_keys = ['game', 'verbs', 'object_names', 'object_nouns', 'object_adjs']

request_step_infos = EnvInfos(
                               description=True,
                               inventory=True,
                               # location=True, # not actually used by qait agent
                               facts=True,
                               last_action=True,
                               admissible_commands=True,
# static infos, don't change during the game:
                               game=True,
                               # verbs=True,
                             )



# info_sample is a skeletal version of info struct returned from gym env, for use with RLPyT (currently not used)
info_sample = {
  "description": "-= Somewhere =-\\nYou are somewhere. There is not much here.",
  "inventory": 'You are carrying nothing.\\n\\n\\n',
  # "location": "somewhere",
  "facts": [('at', ('P', 'P'), ('somewhere', 'r')), ('at', ('not much', 's'), ('somewhere', 'r'))],
  "last_action": None,
  "admissible_commands": ['enter front door', 'examine not much', 'look', 'open front door', 'wait'],
#  "verbs": ['chop', 'close', 'cook', 'dice', 'drink', 'drop', 'eat', 'enter', 'examine', 'go', 'insert', 'inventory',
    #  'lock', 'look', 'open', 'put', 'slice', 'take', 'unlock', 'wait'],
#  # "location_names": ['somewhere', 'elsewhere'],
#  # "location_nouns": ['somewhere', 'elsewhere'],
#  # "location_adjs": ['', '', '', '', '', '', '', '', '', '', ''],
#  "object_names": ['front door', 'plastic gate', 'barn door', 'glass door', 'etc...']
#  "object_nouns": ['door', 'gate', 'door', 'door', 'door', 'gate', 'refrigerator', 'toolbox', 'etc...']
#  "object_adjs": ['front', 'plastic', 'barn', 'glass', 'fiberglass', 'metallic', 'conventional', 'red', 'etc...']
#  "game": "some big nasty string",
#  # "extra.object_locations": {},
#  # "extra.object_attributes": {},
#  "extra.uuid": 'tw-interactive_qa-fAKe-guid'
}


def _serialize_gameinfo(gameinfo_dict) -> str:
    assert 'game' in gameinfo_dict, "GameInfo needs to include a Game object"
    game = gameinfo_dict.pop('game')
    # print(game)
    game_dict = game.serialize()
    print("Serialized game:", game.metadata['uuid'], game_dict.keys())
    print("Game info has:", gameinfo_dict.keys())
    gameinfo_dict['game'] = game_dict
    assert game.metadata['uuid'] == game_dict['metadata']['uuid']
    assert 'extra.uuid' in gameinfo_dict
    assert gameinfo_dict['extra.uuid'] == game_dict['metadata']['uuid']
    assert 'extra.object_locations' in gameinfo_dict
    assert 'extra.object_attributes' in gameinfo_dict
    assert 'object_names' in gameinfo_dict
    assert 'object_nouns' in gameinfo_dict
    assert 'object_adjs' in gameinfo_dict
    _s = json.dumps(gameinfo_dict)
    return _s


def _deserialize_gameinfo(gameinfo_str: str):
    gameinfo_json = json.load(gameinfo_str)
    # TODO: ? maybe deserialize gameinfo_json['game'] -> a Game object
    return gameinfo_json


def get_game_id_from_infos(infos, idx):
    if 'game_id' in infos:
        game_id = parse_gameid(infos['game_id'][idx])
    elif 'extra.uuid' in infos:
        game_id = infos['extra.uuid'][idx]
    else:
        print(f"WARNING: couldn't determine game_id for slot {idx} {len(infos)} {infos.keys()}")
        game_id = None
    return game_id


def parse_gameid(game_id: str) -> str:
    segments = game_id.split('-')
    if len(segments) >= 4:
        code, guid = segments[2:4]
        guid = guid.split('.')[0]
        guid = "{}..{}".format(guid[0:4],guid[-4:])
        segments = code.split('+')
        r, t, g, k, c, o, d = ('0', '0', '0', '_', '_', '_', '_')
        for seg in segments:
            if seg.startswith('recipe'):
                r = seg[len('recipe'):]
            elif seg.startswith('go'):
                g = seg[len('go'):]
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
        shortcode = "r{}t{}{}{}{}{}g{}-{}".format(r,t,k,c,o,d,g,guid)
    else:
        shortcode = game_id
    return shortcode



class QaitEnvWrapper(gym.Wrapper):
    def __init__(self, env, vocab=None, random_seed=None, **kwargs):
        super().__init__(env, **kwargs)
        if random_seed is None:
            random_seed = 42
        self.random_seed = random_seed
        self.vocab = vocab
        self.game_ids = []
        self.tw_oracles = []
        self.episode_counter = -1

    # def _get_gameid(self, idx):
    #     if len(self.game_ids) > idx:
    #         gameid = self.game_ids[idx]
    #     else:
    #         gameid = str(idx)
    #     return gameid

    def set_game_id(self, game_id, idx):
        if idx < len(self.game_ids):
            if game_id != self.game_ids[idx]:
                print(f"[{idx}] {game_id} != {self.game_ids[idx]} NEED TO RESET KG")
                self.game_ids[idx] = game_id
                return True
            else:
                print(f"[{idx}] {game_id} == {self.game_ids[idx]} DON'T RESET KG")
                return False
        else:
            print(f"[{idx}] {game_id} ** NEW AGENT, DON'T RESET: {self.game_ids}")
            assert idx == len(self.game_ids)
            self.game_ids.append(game_id)
            return False

    # def unregister_game(self, env_id):
    #     if env_id in gym.envs.registry.env_specs:
    #         del gym.envs.registry.env_specs[env_id]
    #

    def reset(self, episode_num: Optional[int] = None):
        if episode_num is not None:
            self.episode_counter = episode_num
        else:
            self.episode_counter += 1

        obs, infos = self.env.reset()
        self.vocab.init_from_infos_lists(infos['verbs'], infos['entities'])
        obs, infos = self._on_episode_start(obs, infos, episode_counter=self.episode_counter)
        return obs, infos

    def step(self, commands: List[str]):
        obs, scores, dones, infos = self.env.step(commands)
        if self.tw_oracles:
            assert len(self.tw_oracles) == len(obs)
            for idx, oracle in enumerate(self.tw_oracles):
                if commands and commands[idx] == 'do nothing':
                    pass
                else:
                    oracle.observe(commands[idx], scores[idx], obs[idx], dones[idx], idx=idx)
                    # populate oracle recommended action
                infos = self._compute_oracle_action(idx, infos, obs, dones=dones, verbose=False)
        return obs, scores, dones, infos

    def _compute_oracle_action(self, idx, infos, obs, dones=None, verbose=False):
        if 'tw_o_step' not in infos:
            assert idx == 0, \
                f"we assume idx [{idx}] is enumerating range(len(self.tw_oracles)) [{len(self.tw_oracles)}]"
            infos['tw_o_step'] = ['action'] * len(self.tw_oracles)  # will be replaced before anyone sees these
            if dones and dones[idx]:  # agent idx has already terminated, don't invoke it again
                actiontxt = 'do nothing'
            else:
                actiontxt = self.invoke_oracle(idx, obs[idx], infos, verbose=verbose)
        infos['tw_o_step'][idx] = actiontxt
        return infos

    def _on_episode_start(self, obs: List[str], infos: Dict[str, List[Any]], episode_counter=0):
        # _game_ids = [get_game_id_from_infos(infos, idx) for idx in range(len(obs))]
        # print(f"start_episode[{self.current_episode}] {_game_ids} {self.game_ids}")

        batch_size = len(obs)
        for idx in range(batch_size):
            need_to_forget = True
            game_id = get_game_id_from_infos(infos, idx)
            if game_id:
                need_to_forget = self.set_game_id(game_id, idx)   # if the same game again, remember initial layout
            else:
                game_id = str(idx)
                if idx == len(self.game_ids):
                    self.game_ids.append(game_id)

            if idx == len(self.tw_oracles):
                tw_game_oracle = TextGameAgent(
                        self.random_seed + idx,  #seed
                        "TW",     # rom_name
                        game_id,  # env_name
                        idx=idx,
                        game=None  #TODO: infos['game'][idx]  # for better logging
                )
                self.tw_oracles.append(tw_game_oracle)
                # FIXME: initialization HACK for MEAL
                kg = tw_game_oracle.gi.kg
                meal = kg.create_new_object('meal', MEAL)
                meal.location = kg._nowhere  # the meal doesn't yet exist in the world
                gt = tw_game_oracle.gi.gt
                meal = gt.create_new_object('meal', MEAL)
                meal.location = gt._nowhere  # the meal doesn't yet exist in the world
            else:
                assert idx < len(self.tw_oracles)
                if episode_counter > 0:
                    if need_to_forget:
                        self.tw_oracles[idx].setup_logging(game_id, idx)
                    self.tw_oracles[idx].reset(forget_everything=need_to_forget)
            tw_o = self.tw_oracles[idx]
            assert tw_o.step_num == 0
            _use_groundtruth = False
            task_list = [ExploreHereTask(use_groundtruth=_use_groundtruth),
                         RecipeReaderTask(use_groundtruth=_use_groundtruth)]
            for task in task_list:
                tw_o.task_exec.queue_task(task)

            # populate oracle recommended action
            infos = self._compute_oracle_action(idx, infos, obs, dones=None, verbose=True)
        return obs, infos

    def _on_episode_end(self) -> None:  # NOTE: this is *not* a PL callback
        # Game has finished (either win, lose, or exhausted all the given steps).
        if self.tw_oracles:
            all_endactions = " ".join(
                [":{}: [{}]".format(gameid, tw_oracle.last_action) for idx, (gameid, tw_oracle) in
                 enumerate(zip(self.game_ids, self.tw_oracles))])
            print(f"_end_episode[{self.episode_counter}] <Step {self.tw_oracles[0].step_num}> {all_endactions}")

    def invoke_oracle(self, idx, obstxt, infos, verbose=False) -> str:
        assert idx < len(self.tw_oracles), f"{idx} should be < {len(self.tw_oracles)}"
        tw_oracle = self.tw_oracles[idx]
        # simplify the observation text if it includes notification about incremented score
        if "Your score has just" in obstxt:
            obstxt2 = '\n'.join(
                [line for line in obstxt.split('\n') if not line.startswith("Your score has just")]
            ).strip()
        else:
            obstxt2 = obstxt.strip()
        print("--- current step: {} -- NAIL[{}]: observation=[{}]".format(
            tw_oracle.step_num, idx, obstxt2))
        if 'inventory' in infos:
            print("\tINVENTORY:", infos['inventory'][idx])
        if 'game_id' in infos:
            print("infos[game_id]=", infos['game_id'][idx])
        if 'facts' in infos:

            world_facts = infos['facts'][idx]

            # TODO: remove ground_truth -- no longer needed
            tw_oracle.set_ground_truth(world_facts)

            observable_facts, player_room = filter_observables(world_facts, verbose=verbose)
            print("FACTS IN SCOPE:")
            for fact in observable_facts:
                print('\t', fact)
                # print_fact(game, fact)
        else:
            observable_facts = None

        # if step == 0:    # moved to on_episode_start()
        #     # CHANGED: supply an initial task (read cookbook[prereq location=kitchen]) instead of nav location
        #     # agent.gi.event_stream.push(NeedToDo(RecipeReaderTask(use_groundtruth=agent.gt_nav.use_groundtruth)))
        #     use_groundtruth = False
        #     task_list = [ExploreHereTask(use_groundtruth=use_groundtruth),
        #                  RecipeReaderTask(use_groundtruth=use_groundtruth)]
        #     for task in task_list:
        #         tw_oracle.task_exec.queue_task(task)

        actiontxt = tw_oracle.choose_next_action(obstxt2, observable_facts=observable_facts)
        print("QGYM[{}] choose_next_action -> {}".format(idx, actiontxt))
        return actiontxt


class QaitGym:
    def __init__(self, base_vocab=None, random_seed=42, **kwargs):
        super().__init__(**kwargs)
        self.base_vocab = base_vocab
        self.random_seed = random_seed
        if base_vocab is not None:
            self._action_space = WordSpace(max_length=5, vocab=self.base_vocab.word_vocab)
            self._obs_space = WordSpace(max_length=WordVocab.MAX_NUM_OBS_TOKENS, vocab=self.base_vocab.word_vocab)

    def _register_batch_games(self,
                    gamefiles: List[str],
                    request_infos: Optional[EnvInfos] = None,
                    batch_size: Optional[int] = None,
                    auto_reset: bool = False,
                    max_episode_steps: int = 50,
                    asynchronous: bool = False,
                    action_space: Optional[gym.Space] = None,
                    observation_space: Optional[gym.Space] = None,
                    name: str = None,
                    **kwargs) -> str:

        if request_infos is None:
            request_infos = request_step_infos
        return textworld.gym.register_games(
            gamefiles=gamefiles,
            request_infos=request_infos,
            batch_size=batch_size,
            auto_reset=auto_reset,
            max_episode_steps=max_episode_steps,
            asynchronous=asynchronous,
            action_space=action_space,
            observation_space=observation_space,
            name=name,
            **kwargs)

    def make_batch_env(self, gamefiles, vocab, request_infos: Optional[EnvInfos] = None, batch_size=None, **kwargs):
        if self.base_vocab:
            print("REUSING SHARED WordSpaces")  # NOTE: this HACK doesn't actually speed things up! (unknown why)
            _action_space = self._action_space
            _obs_space = self._obs_space
        else:
            print("CREATING NEW WordSpaces")
            _action_space = WordSpace(max_length=5, vocab=vocab.word_vocab),
            _obs_space = WordSpace(max_length=WordVocab.MAX_NUM_OBS_TOKENS, vocab=vocab.word_vocab)

        batch_env_id = self._register_batch_games(gamefiles=gamefiles,
                                                  request_infos=request_infos,
                                                  batch_size=batch_size,
                                                  asynchronous=False,
 # GVS NOTE: adding the following action_space and observation_space adds some noticeable overhead (close to +3 secs when running 20 eval episodes)
 # TODO: could share across episodes most of the initialization overhead for these gym.Space classes
 # ( especially if we know that wrapped envs will not be used asynchronously )
 #                                                  action_space=Word(max_length=5, vocab=vocab.word_vocab),
 #                                                  observation_space=Word(max_length=WordVocab.MAX_NUM_OBS_TOKENS,
 #                                                                         vocab=vocab.word_vocab),
                                                  action_space=_action_space,
                                                  observation_space=_obs_space,
                                                  **kwargs)
        print(f"Registered {len(gamefiles)} gamefiles as {batch_env_id}")
        for igf, gf in enumerate(gamefiles):
            print(f" *** make_batch_env registered [{igf}]: {gf}")
        #TODO - fix crash: self.rlpyt_env = rlpyt.envs.gym.GymEnvWrapper(env_info, act_null_value='look', obs_null_value='')
        # self.gym_env = rlpyt.envs.gym.make(env_id, info_example=info_sample)
        ## The following lines are more-or-less copied from rlpyt.envs.gym.make()
        gym_env = QaitEnvWrapper(gym.make(batch_env_id), vocab=vocab, random_seed=self.random_seed)
        env_info = rlpyt.envs.gym.EnvInfoWrapper(gym_env, info_sample)
        # #self.rlpyt_env = rlpyt.envs.gym.GymEnvWrapper(env_info)   # this used to crash
        return gym_env  #, batch_env_id

    def ensure_gameinfo_file(self, gamefile, env_seed=42, save_to_file=True):
        # NOTE: individual gamefiles have already been registered
        # NO NEED FOR batch env here
        # if env_id is not None:
        #     assert gamefile == self.env2game_map[env_id]
        print("+++== ensure_gameinfo_file:", gamefile, "...")
        if not _gameinfo_file_exists(gamefile):
            print(f"NEED TO GENERATE game_info: '{_gameinfo_path_from_gamefile(gamefile)}'", )
            print("CURRENT WORKING DIR:", os.getcwd())
            if gamefile.find("game_") > -1:
                game_guid = gamefile[gamefile.find("game_"):].split('_')[1]
            else:
                game_guid = ''
            game_guid += "-ginfo"

            request_qait_infos = EnvInfos(
                # static infos, don't change during the game:
                game=True,
                verbs=True,
                # the following are all specific to TextWorld version customized for QAIT (all are static)
                #UNUSED # location_names=True,
                #UNUSED # location_nouns=True,
                #UNUSED # location_adjs=True,
                #TODO: object_names=True,
                #TODO: object_nouns=True,
                #TODO: object_adjs=True,
                extras=["object_locations", "object_attributes", "uuid"])

            _env = textworld.start(gamefile, infos=request_qait_infos)
            game_state = _env.reset()

            # print(game_state.keys())
            # example from TW 1.3.2 without qait_infos
            # dict_keys(
            #     ['feedback', 'raw', 'game', 'command_templates', 'verbs', 'entities', 'objective', 'max_score', 'extra.seeds',
            #      'extra.goal', 'extra.ingredients', 'extra.skills', 'extra.entities', 'extra.nb_distractors',
            #      'extra.walkthrough', 'extra.max_score', 'extra.uuid', 'description', 'inventory', 'score', 'moves', 'won',
            #      'lost', '_game_progression', '_facts', '_winning_policy', 'facts', '_last_action', '_valid_actions',
            #      'admissible_commands'])
            # game_uuid = game_state['extra.uuid']   # tw-cooking-recipe1+cook+cut+open+drop+go6-xEKyIJpqua0Gsm0q

            game_info = _get_gameinfo_from_gamestate(game_state)  # ? maybe don't filter/ keep everything (including dynamic info)
            _env.close()
            load_additional_gameinfo_from_jsonfile(game_info, _gamejson_path_from_gamefile(gamefile))

            if save_to_file:
                print("+++== save_gameinfo_file:", gamefile, game_info.keys())
                _s = _serialize_gameinfo(game_info)
                with open(_gameinfo_path_from_gamefile(gamefile), "w") as infofile:
                    infofile.write(_s + '\n')
                    infofile.flush()
            game_info['_gamefile'] = gamefile
            return game_info
        else:
            return self.load_gameinfo_file(gamefile)

    def load_gameinfo_file(self, gamefile):
        if not _gameinfo_file_exists(gamefile):
            return self.ensure_gameinfo_file(gamefile)

        game_info = None
        with open(_gameinfo_path_from_gamefile(gamefile), "r") as infofile:
            print("+++== load_gameinfo_file:", gamefile)
            game_info = _deserialize_gameinfo(infofile)
            game_info['_gamefile'] = gamefile
        return game_info


def load_additional_gameinfo_from_jsonfile(gameinfo_dict, filepath):
    with open(_gameinfo_path_from_gamefile(filepath), "r") as jsonfile:
        print("+++== loading gamejson file:", filepath)
        jsondict = json.load(jsonfile)
        print("LOADED:", jsondict.keys())
        for xtra in jsondict['extras']:
            extra_key = "extra."+xtra
            print("\t", extra_key)
            if xtra not in gameinfo_dict:
                gameinfo_dict[extra_key] = jsondict['extras'][xtra]
            else:
                assert jsondict['extras'][xtra] == gameinfo_dict[extra_key],\
                    f"|{jsondict['extras'][xtra]}| SHOULD== |{gameinfo_dict[extra_key]}|"
    return gameinfo_dict


def _gameinfo_path_from_gamefile(gamefile):
    return gamefile.replace(".ulx", ".ginfo")


def _gamejson_path_from_gamefile(gamefile):
    return gamefile.replace(".ulx", ".json")


def _gameinfo_file_exists(gamefile):
    return os.path.exists(_gameinfo_path_from_gamefile(gamefile))


def _get_gameinfo_from_gamestate(gamestate):
    gameinfo = {}
    for key in gamestate:  # keep elements of request_qait_infos and all 'extras'
        if key in request_qait_info_keys or key.startswith("extra"):
            gameinfo[key] = gamestate[key]
        if key == 'game':
            game = gamestate[key]
            gameinfo['object_names'] = game_object_names(game)
            print('   object_names:', gameinfo['object_names'])
            gameinfo['object_nouns'] = game_object_nouns(game)
            print('   object_nouns:', gameinfo['object_nouns'])
            gameinfo['object_adjs'] = game_object_adjs(game)
            print('   object_adjs:', gameinfo['object_adjs'])
    return gameinfo


def _get_gameinfo(infos):
    # serialize infos to json
    filter_out = []
    for key in infos:  # keep elements of request_qait_infos and all 'extras'
        if key not in request_qait_info_keys:
            if not key.startswith("extra"):
                filter_out.append(key)
    for key in filter_out:
        del infos[key]
    return infos


