import os
import json
import textworld
from twutils.twlogic import game_object_names, game_object_nouns, game_object_adjs

request_qait_info_keys = ['game', 'verbs', 'object_names', 'object_nouns', 'object_adjs']


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


def ensure_gameinfo_file(gamefile, env_seed=42, save_to_file=True):
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

        request_qait_infos = textworld.EnvInfos(
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
        return load_gameinfo_file(gamefile)


def load_gameinfo_files(gamefiles):
    game_infos = {}
    # for env_id in env_ids:  # merge multiple infos into one
    #     gamefile = self.qgym.env2game_map[env_id]
    for gamefile in gamefiles:  # merge multiple infos into one
        game_info = load_gameinfo_file(gamefile)  # maybe filter out dynamic info?
        game_uuid = game_info['extra.uuid']
        print("+++ +++ load_gameinfo_file:", gamefile, 'uuid:', game_uuid)
        game_infos[game_uuid] = {}
        for key in game_info:
            if key == '_gamefile':
                game_infos[game_uuid][key] = game_info[key]
            else:
                if not key in game_infos[game_uuid]:
                    game_infos[game_uuid][key] = game_info[key]
                    # game_infos[game_uuid][key] = []
                # game_infos[game_uuid][key].append(game_info[key])
                else:
                    print(game_info)
                    assert False, f"Multiple values for game_infos[{game_uuid}][{key}]"
    return game_infos


def load_gameinfo_file(gamefile):
    if not _gameinfo_file_exists(gamefile):
        return ensure_gameinfo_file(gamefile)

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


