from typing import List
import textworld.EnvInfos

from symbolic.wrappers.gym_wrappers import QaitGym
from symbolic.wrappers.vocab import WordVocab
from .playthroughs import DEFAULT_PTHRU_SEED, MAX_PLAYTHROUGH_STEPS

FTWC_ALL_VOCAB = '~/work2/twdata/ftwc/all-vocab.txt'
QAIT_VOCAB = '~/work2/twdata/qait/qait_word_vocab.txt'

# FTWC_QAIT_VOCAB = '/ssd2tb/ftwc/combined-qait-ftwc-vocab.txt'
# NOTE: there is only one word in all-vocab.txt not in QAIT_VOCAB:  "bbq's"
# (and it would most likely be split during tokenization)


def start_gym_for_playthrough(gamefiles,
                               raw_obs_feedback=True,  # don't apply ConsistentFeedbackWrapper
                               passive_oracle_mode=False,  # if True, don't predict next action
                               max_episode_steps=MAX_PLAYTHROUGH_STEPS,
                               random_seed=DEFAULT_PTHRU_SEED
                               ):  #
    if isinstance(gamefiles, str):
        print("DEPRECATION WARNING: expecting a list of gamefile paths, but got a single str=", gamefiles)
        gamefiles_list = [gamefiles]
    else:
        gamefiles_list = gamefiles
    _word_vocab = WordVocab(vocab_file=None) # QAIT_VOCAB)
    _qgym_ = QaitGym(random_seed=random_seed,
                     raw_obs_feedback=raw_obs_feedback,
                     passive_oracle_mode=passive_oracle_mode)
    _qgym_env = _qgym_.make_batch_env(gamefiles_list,
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
                                   batch_size=len(gamefiles_list),
                                   max_episode_steps=max_episode_steps)
    obs, infos = _qgym_env.reset()
    _word_vocab.init_from_infos_lists(infos['verbs'], infos['entities'])
    return _qgym_env, obs, infos


def step_gym_for_playthrough(gymenv, step_cmds:List[str]):
    obs,rewards,dones,infos = gymenv.step(step_cmds)
    return obs, rewards, dones, infos
