from typing import List, Dict, Optional, Tuple, Mapping, Any
from datetime import timedelta
import numpy as np
import torch

import textworld

from textworld import EnvInfos
from textworld.core import GameState  #, Environment, Wrapper
from textworld.envs.tw import DEFAULT_OBSERVATION

from ..game_agent import TextGameAgent
# from symbolic.event import NeedToAcquire, NeedToGoTo, NeedToDo
from twutils.file_helpers import parse_gameid
from twutils.twlogic import filter_observables, get_recipe
from twutils.twlogic import parse_ftwc_recipe
from twutils.feedback_utils import normalize_feedback_vs_obs_description



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
        print(f"###### TWoWrapper.__init__(random_seed={random_seed}, passive_mode={passive_oracle_mode})")
        if random_seed is None:
            random_seed = 42
        self.random_seed = random_seed
        self.idx = idx   # for more informative logging / debug out when running multiple agents in parallel (vector env)
        self.game_id: Optional[str] = None
        self.tw_oracle: Optional[TextGameAgent] = None
        self.passive_oracle_mode: bool = passive_oracle_mode
        self.episode_counter: int = -1
        self.next_command = None
        self._prev_gamestate: Optional[GameState] = None
        self._initial_gamestate: Optional[GameState] = None
        self.use_internal_names = False

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

    def get_initial_state(self):
        return self._initial_gamestate

    def reset(self):
        print(f"@@@@@@@@@@@ TWoWrapper({self}).reset(env={self._wrapped_env} use_internal_names={self.use_internal_names}")

        # if episode_num is not None:
        #     self.episode_counter = episode_num
        # else:
        #     self.episode_counter += 1

        game_state = self._wrapped_env.reset()
        self._initial_gamestate = game_state
        self._initialize_oracle(game_state, idx=self.idx, forget_everything=False, is_first_episode=True, objective='eat meal')
        # prime the pump
        world_facts = game_state.get('facts', None)
        obstxt = game_state.feedback
        if not hasattr(game_state, 'last_command'):
            print("TWoOracle.reset(): SETTING DEFAULT game_state.last_command = 'start'")
            game_state.last_command = 'start'
        start_action = game_state.last_command
        _actiontxt, _tasks = self.invoke_oracle(obstxt, world_facts, is_done=False, prev_action=start_action, verbose=False)

        # NOTE: _actiontxt and _tasks get stored into self.next_command, self._tasks by .invoke_oracle
        game_state.next_command = self.next_command   # = _actiontxt
        game_state._tasks = _tasks

        if not hasattr(game_state, 'score'):
            game_state.score = 0.0
        game_state.reward = game_state.score
        # print(game_state)
        # obs, infos = self._on_episode_start(obs, infos, episode_counter=self.episode_counter)
        self.episode_counter = -1
        return game_state

    def step(self, command: str):
        self.episode_counter += 1
        #print(f"--------.step({commands})")
        gs, score, done = self._wrapped_env.step(command)
        score = float(score)
        if self._prev_gamestate is not None:
            reward = score - self._prev_gamestate.get('score', 0.0)
        else:
            reward = float(score)
        self._prev_gamestate = gs
        obstxt = gs.feedback
        if obstxt and obstxt.startswith("Invalid"):
            print("WARNING - INVALID COMMAND:", (gs.admissible_commands if hasattr(gs, 'admissible_commands') else 'admissible_commands NOT AVAILABLE'))
        print(f"--------obs:>>{obstxt}<<")
        #TODO: Fix this, this is currently a crude hack!
        if command == "examine cookbook" and obstxt == DEFAULT_OBSERVATION:
            obstxt = get_recipe(self.tw_oracle._game)
        if obstxt is None:
            obstxt = ''

        #??TODO: IS THE FOLLOWING WORKING properly in all cases?
        # ?? Might need to apply ConsistentFeedbackWrapper logic somewhere at a lower level
        new_feedback = normalize_feedback_vs_obs_description(command, obstxt, gs.feedback, gs.description)
        if new_feedback:
            gs.feedback = new_feedback
        obstxt = f"{gs.feedback}\n{gs.inventory if hasattr(gs, 'inventory') else ''}\n{gs.description}"
        if not self.tw_oracle:
            actiontxt = "do something"
            _tasks = ''
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
            actiontxt, _tasks = self.invoke_oracle(obstxt, world_facts, is_done=done, prev_action=prev_action, verbose=False)
        gs.next_command = actiontxt
        gs._tasks = _tasks
        if not hasattr(gs, 'score'):
            gs.score = float(score)
        gs.reward = reward
        return gs, reward, done

    def close(self):
        self._wrapped_env.close()

    def _initialize_oracle(self, game_state, idx=0, objective='eat meal',
                           is_first_episode=True, forget_everything=False):
        game, game_id = get_game_id_from_game_state(game_state)
        is_different_game = self.set_game_id(game_id, idx)  # if playing the same game repeatedly, remember layout
        if not self.tw_oracle or is_different_game or is_first_episode:   # or forget_everything
            self.tw_oracle = TextGameAgent(
                self.random_seed + self.idx,  # seed
                self.game_id,  # env_name
                idx=self.idx,  # for logging
                game=game,      # for more detailed logging
                use_internal_names=self.use_internal_names
            )
        else:      # not is_first_episode
            self.tw_oracle.reset(forget_everything=forget_everything)
        if objective:
            tw_o = self.tw_oracle
            assert tw_o.step_num == 0
            tw_o.set_objective(objective)

    def invoke_oracle(self, obstxt, world_facts, is_done, prev_action=None, verbose=False) -> str:
        if world_facts:
            # TODO:DONE remove ground_truth -- no longer used (but might still be useful for debugging)
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
            obstxt = self.tw_oracle.update_kg(obstxt, observable_facts=observable_facts, prev_action=prev_action)

        if self.passive_oracle_mode or hasattr(self, '_use_oracle') and not self._use_oracle:
            msg = f"--- current step: {self.tw_oracle.step_num} -- TWoWrapper[{self.idx}] passive oracle mode"
            self.tw_oracle.dbg(msg)
            actiontxt = None
            _tasks = ''
        else:
            actiontxt, _tasks = self.get_next_oracle_action(obstxt, is_done)

        self._tasks = _tasks   # remember (used for export to pthru data files)
        self.next_command = actiontxt
        return actiontxt, _tasks

    def get_next_oracle_action(self, obstxt, is_done):
        _tasks = self.tw_oracle.task_exec.tasks_repr()  # a snapshot of oracle state *before* taking next step
        if is_done:
            self.tw_oracle.dbg(f"--- current step: {self.tw_oracle.step_num} is_done=True tasks=[{_tasks}]")
            actiontxt = "do nothing"
        else:
            actiontxt = self.tw_oracle.select_next_action(obstxt, external_next_action=None)
                # actiontxt = tw_oracle.choose_next_action(obstxt, observable_facts=observable_facts)
            self.tw_oracle.dbg(f"--- current step: {self.tw_oracle.step_num} -- TWoWrapper[{self.idx}] choose_next_action -> {actiontxt}")
        return actiontxt,_tasks

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
    # # this version returns values compatible with gym API
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


from twutils.tw_asp_runner import plan_commands_for_game


class TwAspWrapper(TWoWrapper):

    def __init__(self, *args, random_seed: int = None, passive_oracle_mode: bool = False, idx: int = 0, **kwargs):
        super().__init__(*args, random_seed=random_seed, passive_oracle_mode=passive_oracle_mode, **kwargs)
        self._gamepath = None
        self._commands_from_asp = None
        self._next_cmd_from_asp = None  # an iterator, used only if .pthru_cmds have already been supplied (by caller)
        self.pthru_cmds = None
        print(f"###### TwAspWrapper.__init__(random_seed={random_seed}, passive_mode={passive_oracle_mode})")

    def load(self, path: str) -> None:
        """ Loads a new text-based game.
        Arguments:
            path: Path to the game file to load.
        """
        print(f"========= TwAspWrapper.load({path})")
        self._gamepath = path
        self._commands_from_asp = []
        #MOVED TO reset():  self._next_cmd_from_asp = iter(self._commands_from_asp)
        return super().load(path)

    def reset(self):
        print(f"@@@@@@@@@@@ TwAspWrapper({self}).reset(env={self._wrapped_env} use_internal_names={self.use_internal_names}")

        if not self.pthru_cmds:
            if self._use_oracle:
                self._commands_from_asp, self._planner_step_times = plan_commands_for_game(self._gamepath)
            else:
                self._commands_from_asp = []
                self._planner_step_times = None
            self.pthru_cmds = self._commands_from_asp.copy()
            self._next_cmd_from_asp = iter(self._commands_from_asp)
        else:
            self._commands_from_asp = self.pthru_cmds.copy()
            self._next_cmd_from_asp = iter(self._commands_from_asp)

        game_state = super().reset()
        # start_command = next(self._next_cmd_from_asp, None)
        # game_state.last_command = start_command
        original_walkthrough = game_state.get('extra.walkthrough', None)
        print("ORIGINAL WALKTHROUGH:", original_walkthrough)
        if original_walkthrough is not None:
            game_state['extra._walkthrough'] = original_walkthrough
        game_state['extra.walkthrough'] = self._commands_from_asp
        print("WALKTHROUGH from ASP:", game_state.get('extra.walkthrough'))

        return game_state

    def get_next_oracle_action(self, obstxt, is_done):
        _tasks = ''  # a snapshot of oracle state *before* taking next step
        if is_done:
            self.tw_oracle.dbg(f"--- current step (TwAsp): {self.tw_oracle.step_num} is_done=True tasks=[{_tasks}]")
            actiontxt = "do nothing"
        else:
            if self.pthru_cmds:
                actiontxt = next(self._next_cmd_from_asp, None)
                step_num = self.tw_oracle.step_num
                step_time = self.get_twenv_step_time_info(step_num)
                if step_time:
                    print(step_time)
                    elapsed_time = str(timedelta(microseconds=step_time[0]))
                    step_sat = step_time[1]
                else:
                    elapsed_time = None
                    step_sat = True
                log_msg = f"TwAspWrapper: [{step_num-1}] _next_cmd_from_asp: >[ {actiontxt} ]<  solver: {elapsed_time} sat={step_sat}"
                self.tw_oracle.dbg(log_msg)
            else:
                actiontxt = 'do something'
        return actiontxt, _tasks

    # def invoke_oracle(self, obstxt, world_facts, is_done, prev_action=None, verbose=False) -> str:
    #     actiontxt, _tasks = super().invoke_oracle(obstxt, world_facts, is_done, prev_action=prev_action, verbose=verbose)
    #     assert self.passive_oracle_mode
    #     if self.passive_oracle_mode:
    #         if self.pthru_cmds:
    #             actiontxt = next(self._next_cmd_from_asp, None)
    #             step_num = self.tw_oracle.step_num
    #             step_time = self.get_twenv_step_time_info(step_num)
    #             if step_time:
    #                 print(step_time)
    #                 elapsed_time = str(timedelta(microseconds=step_time[0]))
    #                 step_sat = step_time[1]
    #             else:
    #                 elapsed_time = None
    #                 step_sat = True
    #             self.tw_oracle.dbg(f"TwAspWrapper: [{step_num-1}] _next_cmd_from_asp: >[ {actiontxt} ]<  solver: {elapsed_time} sat={step_sat}")
    #         else:
    #             actiontxt = None

    #     self.next_command = actiontxt
    #     return actiontxt, _tasks

    def get_twenv_step_time_info(self, step_num=None):
        if step_num is None:
            step_num = self.tw_oracle.step_num
        step_time = None
        if hasattr(self, "_planner_step_times") and self._planner_step_times is not None:
            if step_num <= len(self._planner_step_times):
                step_time = self._planner_step_times[step_num - 1]
            else:
                print(f"ERROR: index out of range: {step_num} {self._planner_step_times}")
        return step_time

    def copy(self) -> "TwAspWrapper":
        env = TwAspWrapper()
        env._wrapped_env = self._wrapped_env.copy()
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
                               intermediate_reward=True,
                            # policy_commands=True,   # list of commands to win game
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


