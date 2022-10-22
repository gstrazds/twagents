from typing import List, Dict, Optional, Tuple, Mapping, Any
import numpy as np
import torch
import gym

import textworld
import textworld.gym

from textworld import EnvInfos
from textworld.core import GameState  #, Environment, Wrapper

from symbolic.game_agent import TextGameAgent
from symbolic.task_modules import RecipeReaderTask
from symbolic.task_modules.navigation_task import ExploreHereTask
from symbolic.entity import MEAL
# from symbolic.event import NeedToAcquire, NeedToGoTo, NeedToDo
from twutils.file_helpers import parse_gameid
from twutils.twlogic import filter_observables
from twutils.twlogic import parse_ftwc_recipe
from twutils.feedback_utils import normalize_feedback_vs_obs_description
from .vocab import WordVocab


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


class ScoreToRewardWrapper(gym.RewardWrapper):
    """ Converts returned cumulative score into per-step incrmenttal reward.
    Compatible only with vector envs (TW gym env wrapper produces such by default)
    """
    def __init__(self, env):
        super().__init__(env)
        self._prev_score = []

    def reset(self, **kwargs):
        obs, infos = self.env.reset(**kwargs)
        assert isinstance(obs, (list, tuple))   # for use only with vector envs (TW gym env wrapper produces such by default)
        self._prev_score = [0] * len(obs)
        if 'game_score' not in infos:
            infos['game_score'] = self._prev_score
        return obs, infos

    # gym.RewardWrapper
    # def step(self, action):
    #     observation, score, done, info = self.env.step(action)
    #     return observation, self.reward(score), done, info
    def step(self, action):
        observation, score, done, infos = self.env.step(action)
        #if 'game_score' in infos:
        #     assert infos['game_score'] == score, f"{infos['game_score']} should== {score}" #FAILS: infos from prev step
        # else:
        infos['game_score'] = score
        return observation, self.reward(score), done, infos

    def reward(self, score):
        assert isinstance(score, (list, tuple))
        assert len(score) == len(self._prev_score)
        _reward = []
        for _i in range(len(score)):
            _reward.append(score[_i] - self._prev_score[_i])
            self._prev_score[_i] = score[_i]
        return tuple(_reward)


class ConsistentFeedbackWrapper(gym.Wrapper):
    """ Simplifies/normalizes the strings returned in infos['feedback'].
    Compatible only with vector envs (TW gym env wrapper produces such by default)
    """

    def __init__(self, env):
        super().__init__(env)

    # def reset(self, **kwargs):
    #     observation, infos = self.env.reset(**kwargs)
    #     return observation, infos
    def reset(self, **kwargs):
        observation, infos = self.env.reset(**kwargs)
        for idx, obs in enumerate(observation):
            new_feedback = normalize_feedback_vs_obs_description(None,
                    obs, infos['feedback'][idx], infos['description'][idx])
        if new_feedback:
            print(f"MODIFYING infos['feedback'] : '{new_feedback}' <-- orig: {infos['feedback'][idx]}")
            infos['feedback'][idx] = new_feedback
        return observation, infos

    def step(self, action):
        observation, reward, done, infos = self.env.step(action)
        #print(f"ConsistentInfoWrapper: {len(infos['facts'])} {len(observation)} {len(observation[0])}")
        assert isinstance(observation, (list, tuple))   # use with vector envs (TW gym wrapper produces such by default)
        assert 'feedback' in infos, f"infos should include feedback {infos.keys()}"
        assert 'description' in infos, f"infos should include description {infos.keys()}"
        for idx, obs in enumerate(observation):
            new_feedback = normalize_feedback_vs_obs_description(action[idx],
                    obs, infos['feedback'][idx], infos['description'][idx])
            if new_feedback:
                print(f"ConsistenFeedbackWrapper MODIFYING infos['feedback'] : '{new_feedback}' <-- orig:", infos['feedback'][idx])
                infos['feedback'][idx] = new_feedback
            else:
                pass
                #print(f"NOT MODIFYING infos['feedback'] :\n"
                #      f" ----- observation: {obs}\n"
                #      f" ----- infos[feedback]: {infos['feedback'][idx]}\n"
                #      f" ----- infos[description] {infos['description'][idx]}")
        return observation, reward, done, infos


def get_game_id_from_game_state(game_state: GameState):
    game = None
    game_id = None
    if hasattr(game_state, 'game'):
        game = game_state.game
        if hasattr(game, 'metadata'):
            game_id = game.metadata.get('uuid', None)
    return game, game_id


class TWoWrapper(textworld.core.Wrapper):
    """
    Environment wrapper to associate an active env with a TWOracle instance.

    """

    def __init__(self, *args, random_seed: int = None, passive_oracle_mode: bool = False, idx: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        print(f"###### TWoWrapper.__init__(randdom_seed={random_seed}, passive_mode={passive_oracle_mode})")
        if random_seed is None:
            random_seed = 42
        self.random_seed = random_seed
        self.idx = idx   # for more informative logging / debug out when running multiple agents in parallel (vector env)
        self.game_id: Optional[str] = None
        self.tw_oracle: Optional[TextGameAgent] = None
        self.passive_oracle_mode: bool = passive_oracle_mode
        self.episode_counter: int = -1
        self.next_command = None

    # def _wrap(self, env):
    #     super()._wrap(env)

    def set_game_id(self, game_id: str, idx: Optional[int] = None):
        if idx is not None and self.idx is not None:
            assert self.idx == idx
        if self.game_id is None:
            print(f"[{idx}] {game_id} ** NEW AGENT, DON'T RESET: {self.game_id}")
            self.game_id = game_id
            return False
        elif game_id == self.game_id:
            print(f"[{idx}] {game_id} == {self.game_id} DON'T RESET KG")
            return False
        else:  # game_id != self.game_ids:
            print(f"[{idx}] {game_id} != {self.game_id} NEED TO RESET KG")
            self.game_id = game_id
            return True

    def reset(self):
        print(f"@@@@@@@@@@@ TWoWrapper({self}).reset(env={self._wrapped_env}")

        # if episode_num is not None:
        #     self.episode_counter = episode_num
        # else:
        #     self.episode_counter += 1

        game_state = self._wrapped_env.reset()
        self._initialize_oracle(game_state, idx=self.idx, forget_everything=False, is_first_episode=True, objective='eat meal')
        game_state.next_command = self.next_command
        # print(game_state)
        # obs, infos = self._on_episode_start(obs, infos, episode_counter=self.episode_counter)
        self.episode_counter = -1
        return game_state

    def step(self, command: str):
        self.episode_counter += 1
        #print(f"--------.step({commands})")
        gs, reward, done = self._wrapped_env.step(command)
        obstxt = gs.feedback
        print(f"--------obs:>>{obstxt}<<")
        if obstxt is None:
            obstxt = ''
        if not self.tw_oracle:
            actiontxt = "do something"
        else:
            prev_action = command if command else None
            if prev_action == 'do nothing':
                pass
            else:
                self.tw_oracle.observe(reward, obstxt, done, prev_action=prev_action, idx=self.idx)
                # populate oracle recommended action
            if prev_action != "do nothing":
                world_facts = gs.get('facts', None)
            else:
                world_facts = None
            actiontxt, _tasks_ = self.invoke_oracle(obstxt, world_facts, is_done=done, prev_action=prev_action, verbose=False)
        gs.next_command = actiontxt
        return gs, reward, done

    def close(self):
        self._wrapped_env.close()

    def _initialize_oracle(self, game_state, idx=0, is_first_episode=True, forget_everything=False, objective='eat meal'):
        game, game_id = get_game_id_from_game_state(game_state)
        is_different_game = self.set_game_id(game_id, idx)  # if playing the same game repeatedly, remember layout
        if not self.tw_oracle or is_different_game or is_first_episode:   # or forget_everything
            self.tw_oracle = TextGameAgent(
                self.random_seed + self.idx,  # seed
                "TW",  # rom_name
                self.game_id,  # env_name
                idx=self.idx,  # for logging
                game=game      # for more detailed logging
            )
        else:      # not is_first_episode
            self.tw_oracle.reset(forget_everything=forget_everything)
        if objective == 'eat meal':
            tw_o = self.tw_oracle
            assert tw_o.step_num == 0
            _gi = tw_o.gi
            # FIXME: initialization HACK for MEAL
            if not _gi.kg.get_entity('meal'):
                meal = _gi.kg.create_new_object('meal', MEAL)
                _gi.kg._nowhere.add_entity(meal)  # the meal doesn't yet exist in the world
            _use_groundtruth = False
            task_list = [ExploreHereTask(use_groundtruth=_use_groundtruth),
                         RecipeReaderTask(use_groundtruth=_use_groundtruth)]
            for task in task_list:
                tw_o.task_exec.queue_task(task)

        # prime the pump
        world_facts = game_state.get('facts', None)
        obstxt = game_state.feedback
        actiontxt, _tasks = self.invoke_oracle(obstxt, world_facts, is_done=False, prev_action=None, verbose=False)
        return actiontxt, _tasks

    def invoke_oracle(self, obstxt, world_facts, is_done, prev_action=None, verbose=False) -> str:
        _tasks = self.tw_oracle.task_exec.tasks_repr()  # a snapshot of oracle state *before* taking next step
        self._tasks = _tasks   # remember (only used for export to pthru data files)
        # simplify the observation text if it includes notification about incremented score
        if obstxt:
            if "Your score has just" in obstxt:
                obstxt = '\n'.join(
                    [line for line in obstxt.split('\n') if not line.startswith("Your score has just")]
                ).strip()
            else:
                obstxt = obstxt.strip()
        # print(f"--- current step: {tw_oracle.step_num} -- QGYM[{idx}]: observation=[{obstxt}]")
        # if 'inventory' in infos:
        #     print("\tINVENTORY:", infos['inventory'][idx])
        # if 'game_id' in infos:
        #     print("infos[game_id]=", infos['game_id'][idx])

        if world_facts:
            # TODO: remove ground_truth -- no longer needed
            self.tw_oracle.set_ground_truth(world_facts)

            observable_facts, player_room = filter_observables(world_facts, verbose=verbose)
            print("FACTS IN SCOPE:")
            for fact in observable_facts:
                print('\t', fact)
                # print_fact(game, fact)
        else:
            observable_facts = None
        if prev_action != "do nothing":
            # if prev_action=None -> uses oracle._last_action from oracle.observe()
            self.tw_oracle.update_kg(obstxt, observable_facts=observable_facts, prev_action=prev_action)

        if self.passive_oracle_mode:
            print(f"--- current step: {self.tw_oracle.step_num} -- TWoWrapper[{self.idx}] passive oracle mode")
            # return None, _tasks
            actiontxt = None
        elif is_done:
            print(f"--- current step: {self.tw_oracle.step_num} is_done=True: ", _tasks)
            actiontxt = "do nothing"
        else:
            actiontxt = self.tw_oracle.select_next_action(obstxt, external_next_action=None)
            # actiontxt = tw_oracle.choose_next_action(obstxt, observable_facts=observable_facts)
            print(f"--- current step: {self.tw_oracle.step_num} -- TWoWrapper[{self.idx}] choose_next_action -> {actiontxt}")

        self.next_command = actiontxt
        return actiontxt, _tasks






#-------------------------------------
    # def step(self, command: str) -> Tuple[GameState, float, bool]:
    #     gs, reward, done = self._wrapped_env.step(command)
    #     print(f"TWoWrapper.step({command}) -> {reward}, {done}, {gs.keys()}")
    #     return gs, reward, done
    #
    # def reset(self) -> GameState:
    #     gs = self._wrapped_env.reset()
    #     # print(f"TWoWrapper.reset() -> {gs}")
    #     return gs

    # def _get_gamestate_facts(self, game_state: GameState, infos):
    #     infos['extra._facts'] = game_state.get("_facts")
    #     return infos
    #
    # # this version converts to gym API
    # def step(self, command: str) -> Tuple[str, Mapping[str, Any]]:
    #     game_state, score, done = super().step(command)
    #     ob = game_state.feedback
    #     infos = self._get_requested_infos(game_state)
    #     return ob, score, done, infos
    #
    # def reset(self) -> Tuple[str, Mapping[str, Any]]:
    #     game_state = super().reset()
    #     ob = game_state.feedback
    #     infos = self._get_requested_infos(game_state)
    #     return ob, infos

    def copy(self) -> "TWoWrapper":
        env = TWoWrapper()
        env._wrapped_env = self._wrapped_env.copy()
        env.infos = self.infos
        return env


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
    def __init__(self, env, random_seed=None, passive_oracle_mode=False, **kwargs):
        super().__init__(env, **kwargs)
        #print("QaitEnvWrapper.__init__", self.env, self.unwrapped)
        if random_seed is None:
            random_seed = 42
        self.random_seed = random_seed
        self.game_ids = []
        self.tw_oracles = []
        self.passive_oracle_mode = passive_oracle_mode
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
                prev_action = commands[idx] if commands else None
                if prev_action == 'do nothing':
                    pass
                else:
                    oracle.observe(rewards[idx], obs[idx], dones[idx],
                                   prev_action=prev_action, idx=idx)
                    # populate oracle recommended action
                if 'facts' in infos and prev_action != "do nothing":
                    world_facts = infos['facts'][idx]
                else:
                    world_facts = None
                actiontxt, _tasks = self._update_oracle(oracle, idx, obs[idx], world_facts,
                                            is_done=dones[idx], prev_action=prev_action, verbose=False)
                _update_infos(infos, idx, actiontxt, _tasks, len(obs))
        return obs, rewards, dones, infos

    def close(self):
        self.env.close()

    def _update_oracle(self, oracle, idx, obstxt, world_facts, is_done=False, prev_action=None, verbose=False):
        # if is_done:   # if agent idx has already terminated, don't invoke it again
        #     # actiontxt = 'do nothing'
        # else:
        _tasks = oracle.task_exec.tasks_repr()  # a snapshot of oracle state *before* taking next step
        if is_done:
            print("_update_oracle is_done=True: ", _tasks)
        actiontxt = self.invoke_oracle(oracle, idx, obstxt, world_facts, is_done, prev_action=prev_action, verbose=verbose)
        return actiontxt, _tasks

    def _init_oracle(self, game_id, idx=0, need_to_forget=False, is_first_episode=True, objective='eat meal'):
        if idx == len(self.tw_oracles):
            tw_game_oracle = TextGameAgent(
                self.random_seed + idx,  # seed
                "TW",  # rom_name
                game_id,  # env_name
                idx=idx,
                game=None  # TODO: infos['game'][idx]  # for better logging
            )
            self.tw_oracles.append(tw_game_oracle)
        else:
            assert idx < len(self.tw_oracles)
            if not is_first_episode:
                if need_to_forget:
                    self.tw_oracles[idx].setup_logging(game_id, idx)
                self.tw_oracles[idx].reset(forget_everything=need_to_forget)
        if objective == 'eat meal':
            tw_o = self.tw_oracles[idx]
            assert tw_o.step_num == 0
            _gi = tw_o.gi
            # FIXME: initialization HACK for MEAL
            if not _gi.kg.get_entity('meal'):
                meal = _gi.kg.create_new_object('meal', MEAL)
                _gi.kg._nowhere.add_entity(meal)  # the meal doesn't yet exist in the world
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
            if 'facts' in infos:
                world_facts = infos['facts'][idx]
            else:
                world_facts = None
            actiontxt, _tasks = self._update_oracle(self.tw_oracles[idx], idx, obs[idx],
                                        world_facts, is_done=False, verbose=True)
            _update_infos(infos, idx, actiontxt, _tasks, batch_size)
        return obs, infos

    def _on_episode_end(self) -> None:  # NOTE: this is *not* a PL callback
        # Game has finished (either win, lose, or exhausted all the given steps).
        if self.tw_oracles:
            all_endactions = " ".join(
                [":{}: [{}]".format(gameid, tw_oracle.last_action) for idx, (gameid, tw_oracle) in
                 enumerate(zip(self.game_ids, self.tw_oracles))])
            print(f"_end_episode[{self.episode_counter}] <Step {self.tw_oracles[0].step_num}> {all_endactions}")

    def invoke_oracle(self, tw_oracle, idx, obstxt, world_facts, is_done, prev_action=None, verbose=False) -> str:
        assert idx < len(self.tw_oracles), f"{idx} should be < {len(self.tw_oracles)}"
        assert tw_oracle == self.tw_oracles[idx]
        # simplify the observation text if it includes notification about incremented score
        if obstxt:
            if "Your score has just" in obstxt:
                obstxt = '\n'.join(
                    [line for line in obstxt.split('\n') if not line.startswith("Your score has just")]
                ).strip()
            else:
                obstxt = obstxt.strip()
        # print(f"--- current step: {tw_oracle.step_num} -- QGYM[{idx}]: observation=[{obstxt}]")
        # if 'inventory' in infos:
        #     print("\tINVENTORY:", infos['inventory'][idx])
        # if 'game_id' in infos:
        #     print("infos[game_id]=", infos['game_id'][idx])

        if world_facts:

            # TODO: remove ground_truth -- no longer needed
            tw_oracle.set_ground_truth(world_facts)

            observable_facts, player_room = filter_observables(world_facts, verbose=verbose)
            print("FACTS IN SCOPE:")
            for fact in observable_facts:
                print('\t', fact)
                # print_fact(game, fact)
        else:
            observable_facts = None
        if prev_action != "do nothing":
            # if prev_action=None -> uses oracle._last_action from oracle.observe()
            tw_oracle.update_kg(obstxt, observable_facts=observable_facts, prev_action=prev_action)

        if self.passive_oracle_mode:
            print(f"--- current step: {tw_oracle.step_num} -- QGYM[{idx}] passive oracle mode")
            return None
        if is_done:
            actiontxt = "do nothing"
        else:
            actiontxt = tw_oracle.select_next_action(obstxt, external_next_action=None)
            # actiontxt = tw_oracle.choose_next_action(obstxt, observable_facts=observable_facts)
            print(f"--- current step: {tw_oracle.step_num} -- QGYM[{idx}] choose_next_action -> {actiontxt}")
        return actiontxt


def _update_infos(infos, idx, actiontxt, taskstack, batch_size):
    if actiontxt:
        if 'tw_o_step' not in infos:
            assert idx == 0, \
                f"if tw_o_step is missing, we assume idx [{idx}] is enumerating range(len(self.tw_oracles)) [{batch_size}]"
            infos['tw_o_step'] = ['fake action'] * batch_size  # will be replaced before anyone sees these
        infos['tw_o_step'][idx] = actiontxt

    if 'tw_o_stack' not in infos:
        assert idx == 0, \
            f"if tw_o_stack is missing, we assume idx [{idx}] is enumerating range(len(self.tw_oracles)) [{batch_size}]"
        infos['tw_o_stack'] = ['(( )) [[ task ]]'] * batch_size  # will be replaced before anyone sees these
    infos['tw_o_stack'][idx] = taskstack


class QaitGym:
    def __init__(self, base_vocab=None, random_seed=42,
                 raw_obs_feedback=False,  # if true, don't apply ConsistentFeedbackWrapper
                 passive_oracle_mode=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.base_vocab = base_vocab
        self.random_seed = random_seed
        self.raw_obs_feedback_mode = raw_obs_feedback
        self.passive_oracle_mode=passive_oracle_mode
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
                                                #   action_space=_action_space,
                                                #   observation_space=_obs_space,
                                                  max_episode_steps=max_episode_steps,
                                                  **kwargs)
        print(f"Registered {len(gamefiles)} gamefiles as {batch_env_id}")
        for igf, gf in enumerate(gamefiles):
            print(f" *** make_batch_env registered [{igf}]: {gf}")
        base_env = gym.make(batch_env_id)
        wrapped_env = ScoreToRewardWrapper(base_env)
        if request_infos and request_infos.feedback and not self.raw_obs_feedback_mode:
            wrapped_env = ConsistentFeedbackWrapper(wrapped_env)
        else:
            if request_infos and request_infos.feedback:
                print("QaitGym.raw_obs_feedback_mode -- Skipping ConsistentFeedbackWrapper")
            else:
                print("WARNING: QaitGym -- Skipping ConsistentFeedbackWrapper because request_infos.feedback is not set")

        gym_env = QaitEnvWrapper(wrapped_env, random_seed=self.random_seed, passive_oracle_mode=self.passive_oracle_mode)
        return gym_env  #, batch_env_id


