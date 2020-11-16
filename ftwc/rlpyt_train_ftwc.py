import os
import pathlib
from os.path import join as pjoin
import json
import datetime

import numpy as np

import hydra
from omegaconf import OmegaConf, DictConfig
import rlpyt
import gym

from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.algos.dqn.dqn import DQN
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.utils.logging.context import logger_context

# from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
# from rlpyt.samplers.parallel.gpu.collectors import (GpuResetCollector,
#     GpuWaitResetCollector)
from rlpyt.envs.atari.atari_env import AtariEnv, AtariTrajInfo
from rlpyt.agents.dqn.atari.atari_dqn_agent import AtariDqnAgent
# from rlpyt.algos.pg.a2c import A2C
# from rlpyt.agents.pg.atari import AtariLstmAgent
# from rlpyt.runners.minibatch_rl import MinibatchRl
# from rlpyt.utils.logging.context import logger_context

from textworld.gym import register_games
from textworld.generator.game import Game

# from game_generator import request_infos, request_game_infos, ensure_gameinfo_file, GameSource
# from agent import Agent

def build_and_train(cfg, game="ftwc", run_ID=0):
    #GVS NOTE: for ftwc/qait ?use CpuWaitResetCollector  (or CpuResetCollector)
    sampler = SerialSampler(
        EnvCls=AtariEnv,
        TrajInfoCls=AtariTrajInfo,  # default traj info + GameScore
        env_kwargs=dict(game=game),
        eval_env_kwargs=dict(game=game),
        batch_T=4,  # Four time-steps per sampler iteration.
        batch_B=1,
        max_decorrelation_steps=0,
        eval_n_envs=10,
        eval_max_steps=int(10e2),
        eval_max_trajectories=5,
    )
    algo = DQN(min_steps_learn=1e2)  # Run with defaults.
    agent = AtariDqnAgent()
    runner = MinibatchRlEval(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=50e6,
        log_interval_steps=1e3,
        affinity=dict(cuda_idx=cfg.cuda_idx),
    )
    config = dict(game=game)
    name = "dqn_" + game
    log_dir = "ftwc"
    with logger_context(log_dir, run_ID, name, config, snapshot_mode="last"):
        runner.train()

def _choose_games_from_directory(dirpath, num_games=-1):
    games_list = [str(path) for path in pathlib.Path(dirpath).glob('*.ulx')]
    if num_games > 0:
        sampled_games = np.random.choice(games_list, num_games, replace=False).tolist()
    else:
        sampled_games = games_list
    # print("_________ _choose_games_from_directory() -- from", len(games_list), "selected:\n", "\n".join(sampled_games))
    return sampled_games

# from agent import info_sample

from rlpyt.envs.gym_schema import GymEnvWrapper, EnvInfoWrapper

NUM_GAMES = 3

@hydra.main(config_path="conf", config_name="ftwc_rlpyt")
def main(cfg):
    cfg.cwd_path = hydra.utils.to_absolute_path(cfg.cwd_path)
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print("cwd_path = ", cfg.cwd_path)
    start_time = datetime.datetime.now()
    print(f"=============================================== {os.path.basename(__file__)} - Start time:", start_time)
    print(os.getcwd(), '\n')

    build_and_train(cfg)
    # build_and_train(**cfg.example_4)
    #     game=cfg.game,
    #     run_ID=cfg.run_ID,
    #     cuda_idx=cfg.cuda_idx,
    #     mid_batch_reset=cfg.mid_batch_reset,
    #     n_parallel=cfg.n_parallel,
    # )

    # foobar(cfg)

    finish_time = datetime.datetime.now()
    print(f"=================================================== {os.path.basename(__file__)} - Finished :"
          f" {finish_time} -- elapsed: {finish_time-start_time}")


def foobar(cfg):
    pass
    # games = _choose_games_from_directory(cfg.training.pregenerated_games, num_games=NUM_GAMES)
    # print("Selected games:", games)
    # batch_env_ids = []
    # for game in games:
    #     _env_id = register_game(game, request_infos=request_infos)
    #     print(_env_id)
    #     batch_env_ids.append(_env_id)
    #     game_info = ensure_gameinfo_file(game, save_to_file=False)
    #     # for key in game_info:
    #     #     if key != 'game':
    #     #         print("-->>> ", key, ":\t", game_info[key])
    #     if 'game' in game_info:
    #         game = game_info['game']
    #         print("\n------- Serialized game: -------", )
    #         if 'metadata' in game and 'uuid' in game['metadata']:
    #             print('uuid:', game['metadata']['uuid'], end=' ')
    #         print([k for k in game], end=" ")
    #         if 'extras' in game:
    #             print("extras:", [x for x in game['extras']], end=' ')
    #             # if 'object_locations' in game['extras']:
    #             #     print("object_locations=", game['extras']['object_locations'])
    #         print()
    #         if 'objective' in game and game['objective']:
    #             print("game.objective:", game['objective'])
    #
    # batch_env_id = make_batch2(batch_env_ids)

    # rlpyt_env = rlpyt.envs.gym_schema.make(batch_env_id, info_example=info_sample)
    ## EQUIV TO:
    ## gym_env = gym.make(_env_id)
    ## print("DEBUG: gym_env.action_space", gym_env.action_space)
    ## print("DEBUG: gym_env.observation_space", gym_env.observation_space)
    ## NEXT LINE HACK NO LONGER NEEDED BECAUSE OF HACK to rlpyt.envs.gym_schema.py
    #### gym_env.action_space.sample = lambda: "look"  # monkey patch to avoid a crash
    ## env_info = EnvInfoWrapper(gym_env, info_sample)
    ## rlpyt_env = rlpyt.envs.gym_schema.GymEnvWrapper(env_info)   # this (used to) crash (fixed by hacking rlptyt)

    #obs, reward, done, infos = rlpyt_env.step(["go north"] * NUM_GAMES)
    #print(obs)
    ## print(infos)
    ## print(infos['extra.uuid'])
    #print(infos.location)
    #    print(infos.last_action)
    ## print(infos.admissible_commands)

    # for gamestr in infos.game:
    #     game = Game.deserialize(gamestr)
    #     print("-----------------------------------", game.metadata['uuid'])
    #     print(game.extras['uuid'])
    #     print(game.extras['object_locations'])


if __name__ == "__main__":
    main()
