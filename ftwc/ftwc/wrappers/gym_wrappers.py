import gym
from typing import List, Dict, Optional, Any
import numpy as np
import torch

import textworld
import textworld.gym
from textworld import EnvInfos
import rlpyt.envs.gym

from symbolic.game_agent import TextGameAgent
from symbolic.task_modules import RecipeReaderTask
from symbolic.task_modules.navigation_task import ExploreHereTask
from symbolic.entity import MEAL
# from symbolic.event import NeedToAcquire, NeedToGoTo, NeedToDo
from twutils.twlogic import filter_observables
from twutils.gym_wrappers import ScoreToRewardWrapper, ConsistentFeedbackWrapper
from ftwc.vocab import WordVocab


class MultiWordSpace(gym.spaces.MultiDiscrete):  # adapted from textworld.gym.text_spaces.WordSpace
    """ Word observation/action space

    This space consists of a series of `gym.spaces.Discrete` objects all with
    the same parameters. Each `gym.spaces.Discrete` can take integer values
    between 0 and `len(self.vocab)`.

    Notes
    -----
    The following special tokens will be prepended (if needed) to the vocabulary:
    <PAD> : Padding
    <UNK> : Unknown word
    <S>   : Beginning of sentence
    </S>  : End of sentence
    """

    def __init__(self, max_length, vocab: WordVocab, cmd_phrases=False):
        """
        Parameters
        ----------
        max_length : int
            Maximum number of words in a text.
        vocab :
            Vocabulary defining this space. It shouldn't contain any
            duplicate words.
        """
        # if len(vocab) != len(set(vocab)):
        #     raise VocabularyHasDuplicateTokens()
        self._is_cmd_phrases = cmd_phrases
        self.vocab = vocab
        self.vocab_size = vocab.vocab_size
        self.max_length = max_length
        super().__init__([self.vocab_size] * self.max_length)
        self.dtype = np.int64  # Overwrite Gym's dtype=int8.

    def sample(self):
        if self._is_cmd_phrases:
            return self.vocab.generate_random_command_phrase(None)
        else:
            return super().sample()

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

request_step_infos = EnvInfos(
                               description=True,
                               inventory=True,
                               feedback=True,
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


def get_game_id_from_infos(infos, idx):
    if 'game_id' in infos:
        game_id = parse_gameid(infos['game_id'][idx])
    elif 'extra.uuid' in infos:
        game_id = infos['extra.uuid'][idx]
    else:
        print(f"WARNING: couldn't determine game_id for slot {idx} {len(infos)} {infos.keys()}")
        game_id = None
    return game_id



class QaitEnvWrapper(gym.Wrapper):
    def __init__(self, env, random_seed=None, **kwargs):
        super().__init__(env, **kwargs)
        #print("QaitEnvWrapper.__init__", self.env, self.unwrapped)
        if random_seed is None:
            random_seed = 42
        self.random_seed = random_seed
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
        obs, infos = self._on_episode_start(obs, infos, episode_counter=self.episode_counter)
        return obs, infos

    def step(self, commands: List[str]):
        #print(f"--------QAIT WRAPPER.step({commands})")
        obs, rewards, dones, infos = self.env.step(commands)
        #print(f"--------QAIT WRAPPER => obs:>>{obs}<<")
        if self.tw_oracles:
            assert len(self.tw_oracles) == len(obs)
            for idx, oracle in enumerate(self.tw_oracles):
                if commands and commands[idx] == 'do nothing':
                    pass
                else:
                    oracle.observe(rewards[idx], obs[idx], dones[idx], prev_action=None, idx=idx)  #UNNEEDED, causes problems: prev_action=commands[idx]
                    # populate oracle recommended action
                infos = self._compute_oracle_action(idx, infos, obs, dones=dones, verbose=False)
        return obs, rewards, dones, infos

    def _compute_oracle_action(self, idx, infos, obs, dones=None, verbose=False):
        if 'tw_o_step' not in infos:
            assert idx == 0, \
                f"if tw_o_step is missing, we assume idx [{idx}] is enumerating range(len(self.tw_oracles)) [{len(self.tw_oracles)}]"
            infos['tw_o_step'] = ['fake action'] * len(obs)  # will be replaced before anyone sees these
        if dones and dones[idx]:  # agent idx has already terminated, don't invoke it again
            actiontxt = 'do nothing'
        else:
            actiontxt = self.invoke_oracle(idx, obs[idx], infos, verbose=verbose)
        infos['tw_o_step'][idx] = actiontxt
        return infos

    def _init_oracle(self, game_id, idx=0, need_to_forget=False, is_first_episode=True):
        if idx == len(self.tw_oracles):
            tw_game_oracle = TextGameAgent(
                self.random_seed + idx,  # seed
                "TW",  # rom_name
                game_id,  # env_name
                idx=idx,
                game=None  # TODO: infos['game'][idx]  # for better logging
            )
            self.tw_oracles.append(tw_game_oracle)
            # FIXME: initialization HACK for MEAL
            kg = tw_game_oracle.gi.kg
            meal = kg.create_new_object('meal', MEAL)
            kg._nowhere.add_entity(meal)  # the meal doesn't yet exist in the world
            gt = tw_game_oracle.gi.gt
            meal = gt.create_new_object('meal', MEAL)
            gt._nowhere.add_entity(meal)  #meal.location = gt._nowhere  # the meal doesn't yet exist in the world
        else:
            assert idx < len(self.tw_oracles)
            if not is_first_episode:
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

    def _on_episode_start(self, obs: List[str], infos: Dict[str, List[Any]], episode_counter=0):
        # _game_ids = [get_game_id_from_infos(infos, idx) for idx in range(len(obs))]
        # print(f"start_episode[{self.current_episode}] {_game_ids} {self.game_ids}")

        batch_size = len(obs)
        for idx in range(batch_size):
            game_id = get_game_id_from_infos(infos, idx)
            if game_id:
                need_to_forget = self.set_game_id(game_id, idx)   # if playing the same game repeatedly, remember layout
            else:
                need_to_forget = False
                game_id = str(idx)
                if idx == len(self.game_ids):
                    self.game_ids.append(game_id)
            self._init_oracle(game_id, idx, need_to_forget=need_to_forget, is_first_episode=(episode_counter == 0))
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
        print(f"--- current step: {tw_oracle.step_num} -- QGYM[{idx}]: observation=[{obstxt2}]")
        if 'inventory' in infos:
            print("\tINVENTORY:", infos['inventory'][idx])
        if 'game_id' in infos:
            print("infos[game_id]=", infos['game_id'][idx])
        if 'facts' in infos:

            world_facts = infos['facts'][idx]

            # if world_facts:   # facts can be a list of Proposition or serialized (json) Propositions
            #     if isinstance(world_facts[0], textworld.logic.Proposition):
            #         world_facts = [f.serialize() for f in world_facts]

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
        print(f"--- current step: {tw_oracle.step_num} -- QGYM[{idx}] choose_next_action -> {actiontxt}")
        return actiontxt


class QaitGym:
    def __init__(self, base_vocab=None, random_seed=42, **kwargs):
        super().__init__(**kwargs)
        self.base_vocab = base_vocab
        self.random_seed = random_seed
        if base_vocab is not None:
            self._action_space = MultiWordSpace(max_length=5, vocab=self.base_vocab, cmd_phrases=True)
            self._obs_space = MultiWordSpace(max_length=WordVocab.MAX_NUM_OBS_TOKENS, vocab=self.base_vocab)

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

    def make_batch_env(self, gamefiles, vocab, request_infos: Optional[EnvInfos] = None,
                       batch_size=None, max_episode_steps=100, **kwargs):
        if self.base_vocab:
            print("REUSING SHARED WordSpaces")  # NOTE: this HACK doesn't actually speed things up! (unknown why)
            _action_space = self._action_space
            _obs_space = self._obs_space
        else:
            print("CREATING NEW WordSpaces")
            _action_space = MultiWordSpace(max_length=5, vocab=vocab, cmd_phrases=True),
            _obs_space = MultiWordSpace(max_length=WordVocab.MAX_NUM_OBS_TOKENS, vocab=vocab)

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
                                                  max_episode_steps=max_episode_steps,
                                                  **kwargs)
        print(f"Registered {len(gamefiles)} gamefiles as {batch_env_id}")
        for igf, gf in enumerate(gamefiles):
            print(f" *** make_batch_env registered [{igf}]: {gf}")
        #TODO - fix crash: self.rlpyt_env = rlpyt.envs.gym.GymEnvWrapper(env_info, act_null_value='look', obs_null_value='')
        # self.gym_env = rlpyt.envs.gym.make(env_id, info_example=info_sample)
        ## The following lines are more-or-less copied from rlpyt.envs.gym.make()
        base_env = gym.make(batch_env_id)
        wrapped_env = ScoreToRewardWrapper(base_env)
        if request_infos and request_infos.feedback:
            wrapped_env = ConsistentFeedbackWrapper(wrapped_env)
        else:
            print("WARNING: skipping ConsistentFeedbackWrapper because request_infos.feedback is not set")

        gym_env = QaitEnvWrapper(wrapped_env, random_seed=self.random_seed)
        env_info = rlpyt.envs.gym.EnvInfoWrapper(gym_env, info_sample)
        # #self.rlpyt_env = rlpyt.envs.gym.GymEnvWrapper(env_info)   # this used to crash
        return gym_env  #, batch_env_id


