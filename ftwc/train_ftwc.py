import os
import glob
import argparse

from typing import List, Dict, Tuple, Any, Optional

import datetime
import hydra
from omegaconf import OmegaConf, DictConfig


from pytorch_lightning import Trainer, seed_everything

from tqdm import tqdm

from textworld import EnvInfos

from ftwc_agent import FtwcAgent, WordVocab
from ftwc_data import GamefileDataModule

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


def train(cfg, dataloader):
    vocab = WordVocab(vocab_file=cfg.general.vocab_words)
    agent = FtwcAgent(cfg, vocab=vocab)
    # requested_infos = agent.select_additional_infos()
    # _validate_requested_infos(requested_infos)

    for epoch_no in range(1, cfg.training.nb_epochs + 1):
        # # start fresh for each epoch
        # # (GVS Oct-05-2020: this was a hack to work around bugs in resetting kg to initial state)
        # if epoch_no > 1:
        #     agent = CustomAgent()
        stats = {
            "scores": [],
            "steps": [],
        }
        # for game_no in tqdm(range(len(game_files))):
        #     agent.train()  # agent in training mode
        #     gamefile = game_files[game_no]
        for batch in dataloader:
            for gamefile in batch:
                print(f"gamefile = {gamefile}")
                scores, steps = agent.run_episode(gamefile)
                stats["scores"].extend(scores)
                stats["steps"].extend(steps)

        score = sum(stats["scores"]) / agent.batch_size
        steps = sum(stats["steps"]) / agent.batch_size
        print("Epoch: {:3d} | {:2.1f} pts | {:4.1f} steps".format(epoch_no, score, steps))


@hydra.main(config_path="conf", config_name="ftwc")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.general.random_seed)
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

    test_games = []
    for game in cfg.test.games:
        game = os.path.expanduser(game)
        if os.path.isdir(game):
            test_games += glob.glob(os.path.join(game, "*.ulx"))
        else:
            test_games.append(game)

    print("{} games found for training.".format(len(games)))
    # print(games)
    data = GamefileDataModule(gamefiles=games, testfiles=test_games)
    # data.setup()
    # train(cfg, data.test_dataloader())

    vocab = WordVocab(vocab_file=cfg.general.vocab_words)
    agent = FtwcAgent(cfg, vocab=vocab)
    trainer = Trainer(
        gpus=1,
        deterministic=True,
        early_stop_callback=False,
        # distributed_backend='dp',
        # val_check_interval=100,
        max_epochs=0,  # does not call training_step() at all
        limit_val_batches=0,  # prevent validation_step() from getting called
        limit_test_batches=cfg.test.limit_test_batches  #speed things up for debugging
    )
    trainer.fit(agent, data)
    trainer.test(datamodule=data)
    # trainer.fit(agent, data)
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
