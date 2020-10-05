import os
import glob
import argparse

import datetime
import hydra
from omegaconf import OmegaConf, DictConfig

from tqdm import tqdm

import gym
import textworld.gym
from textworld import EnvInfos

from ftwc_agent import CustomAgent, WordVocab

# List of additional information available during evaluation.
AVAILABLE_INFORMATION = EnvInfos(
    description=True,
    inventory=True,
    max_score=True,
    objective=True,
    entities=True,
    verbs=True,
    command_templates=True,
    admissible_commands=True,
    won=True,
    lost=True,
    extras=["recipe", "uuid"]
)


def _validate_requested_infos(infos: EnvInfos):
    msg = "The following information cannot be requested: {}"
    for key in infos.basics:
        if not getattr(AVAILABLE_INFORMATION, key):
            print(msg.format(key))
            # raise ValueError(msg.format(key))

    for key in infos.extras:
        if key not in AVAILABLE_INFORMATION.extras:
            print(msg.format(key))
            # raise ValueError(msg.format(key))


def train(cfg, game_files):
    vocab = WordVocab(vocab_file=cfg.general.vocab_words)
    agent = CustomAgent(cfg, vocab=vocab)
    requested_infos = agent.select_additional_infos()
    _validate_requested_infos(requested_infos)
    requested_infos.facts = True  # use ground truth facts about the world (this is a training oracle)

    for epoch_no in range(1, agent.nb_epochs + 1):
        # # start fresh for each epoch
        # # TODO: (GVS Oct-02-2020: this is a hack to work around bugs in resetting kg to initial state)
        # if epoch_no > 1:
        #     agent = CustomAgent()
        stats = {
            "scores": [],
            "steps": [],
        }
        for game_no in tqdm(range(len(game_files))):
            agent.train()  # agent in training mode
            gamefile = game_files[game_no]
            env_id = textworld.gym.register_games([gamefile],
                                                  requested_infos,
                                                  max_episode_steps=agent.max_nb_steps_per_episode,
                                                  batch_size=agent.batch_size,
                                                  asynchronous=True,
                                                  # auto_reset=auto_reset,
                                                  # action_space=action_space,
                                                  # observation_space=observation_space,
                                                  name="training")
            # env_id = textworld.gym.make_batch(env_id, batch_size=agent.batch_size, parallel=True)
            env = gym.make(env_id)
            obs, infos = env.reset()

            scores = [0] * len(obs)
            dones = [False] * len(obs)
            steps = [0] * len(obs)
            while not all(dones):
                # Increase step counts.
                steps = [step + int(not done) for step, done in zip(steps, dones)]
                commands = agent.act(obs, scores, dones, infos)
                obs, scores, dones, infos = env.step(commands)

            # Let the agent knows the game is done.
            agent.act(obs, scores, dones, infos)

            stats["scores"].extend(scores)
            stats["steps"].extend(steps)

        score = sum(stats["scores"]) / agent.batch_size
        steps = sum(stats["steps"]) / agent.batch_size
        print("Epoch: {:3d} | {:2.1f} pts | {:4.1f} steps".format(epoch_no, score, steps))


@hydra.main(config_path="conf", config_name="ftwc")
def main(cfg: DictConfig) -> None:
    cfg.cwd_path = hydra.utils.to_absolute_path(cfg.cwd_path)
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print("cwd_path = ", cfg.cwd_path)
    start_time = datetime.datetime.now()
    print("=================================================== evaluate.py - Start time:", start_time)
    print(os.getcwd(), '\n')

    games = []
    for game in cfg.training.games:
        game = os.path.expanduser(game)
        if os.path.isdir(game):
            games += glob.glob(os.path.join(game, "*.ulx"))
        else:
            games.append(game)

    print("{} games found for training.".format(len(games)))
    # print(games)
    train(cfg, games)

    # model = DQNLightning(**cfg)
    #
    # trainer = pl.Trainer(
    #     gpus=1,
    #     distributed_backend='dp',
    #     early_stop_callback=False,
    #     val_check_interval=100
    # )
    #
    # trainer.fit(model)
    finish_time = datetime.datetime.now()
    print(f"=================================================== evaluate.py - Finished : {finish_time} -- elapsed: {finish_time-start_time}")


if __name__ == '__main__':
    main()

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="Train an agent.")
#     parser.add_argument("games", metavar="game", nargs="+",
#                        help="List of games (or folders containing games) to use for training.")
#     args = parser.parse_args()
#
#     games = []
#     for game in args.games:
#         if os.path.isdir(game):
#             games += glob.glob(os.path.join(game, "*.ulx"))
#         else:
#             games.append(game)
#
#     print("{} games found for training.".format(len(games)))
#     train(games)
#

