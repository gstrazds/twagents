import os
from typing import List, Dict, Tuple, Any, Optional
from collections import OrderedDict

import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim.optimizer import Optimizer

import pytorch_lightning as pl

import gym
from textworld import EnvInfos
import textworld.gym

from .model import LSTM_DQN
from .generic import to_np, to_pt, pad_sequences, max_len
from .buffers import HistoryScoreCache, PrioritizedReplayMemory, Transition
from .vocab import WordVocab
from .wrappers import QaitGym, ScoreToRewardWrapper


def _choose_random_command(_unused_word_ranks_, word_masks_np, use_cuda):
    """
    Generate a command randomly, for epsilon greedy.

    Arguments:
        word_ranks: Q values for each word by model.action_scorer.
        word_masks_np: Vocabulary masks for words depending on their type (verb, adj, noun, adj2, noun2).
    """
    # assert len(word_ranks) == len(word_masks_np)

    # word_ranks_np = [to_np(item) for item in word_ranks]  # list of (batch x n_vocab) arrays, len=5 (5 word output phrases)
    # # GVS QUESTION? why is this next line commented out here? ( compare _choose_maxQ_command() )
    # # word_ranks_np = [r - np.min(r) for r in word_ranks_np]  # minus the min value, so that all values are non-negative
    # # GVS ANSWER: because the values in word_ranks_np are never actually used
    # word_ranks_np = [r * m for r, m in zip(word_ranks_np, word_masks_np)]  # list of batch x n_vocab

    # batch_size = word_ranks[0].size(0)
    # print("batch_size=", batch_size, len(word_masks_np))
    word_indices = []
    for i in range(len(word_masks_np)):  # len=5 (verb, adj1, noun1, adj2, noun2)
        indices = []
        # for j in range(batch_size):
        #     msk = word_masks_np[i][j]  # msk is of len = vocab, j is index into batch
        for msk in word_masks_np[i]:
            indices.append(np.random.choice(len(msk), p=msk / np.sum(msk, -1)))  # choose from non-zero entries of msk
        word_indices.append(np.array(indices))
    # word_indices: list of batch

    # word_qvalues = [[] for _ in word_masks_np]
    # for i in range(batch_size):
    #     for j in range(len(word_qvalues)):
    #         word_qvalues[j].append(word_ranks[j][i][word_indices[j][i]])
    # word_qvalues = [torch.stack(item) for item in word_qvalues]
    word_indices = [to_pt(item, use_cuda) for item in word_indices]
    word_indices = [item.unsqueeze(-1) for item in word_indices]  # list of 5 tensors, each w size: batch x 1
    # return word_qvalues, word_indices
    return word_indices


def _choose_maxQ_command(word_ranks, word_masks_np, use_cuda):
    """
    Generate a command by maximum q values, for epsilon greedy.

    Arguments:
        word_ranks: Q values for each word by model.action_scorer.
        word_masks_np: Vocabulary masks for words depending on their type (verb, adj, noun).
    """
    word_ranks_np = [to_np(item) for item in word_ranks]  # list of arrays, batch_len x n_vocab
    word_ranks_np = [r - np.min(r) for r in word_ranks_np]  # minus the min value, so that all values are non-negative
    word_ranks_np = [r * m for r, m in zip(word_ranks_np, word_masks_np)]  # list of batch x n_vocab

    word_indices = [np.argmax(item, -1) for item in word_ranks_np]  # list of 5 arrays, each w len = batch

    # word_qvalues = [[] for _ in word_masks_np]
    # batch_size = word_ranks[0].size(0)
    # # print("batch_size=", batch_size, "len(word_qvalues)=", len(word_qvalues)) #batch_size=[16 or 24], len(word_qvalues)=5.
    # for i in range(batch_size):
    #     for j in range(len(word_qvalues)):
    #         word_qvalues[j].append(word_ranks[j][i][word_indices[j][i]])
    # word_qvalues = [torch.stack(item) for item in word_qvalues]
    word_indices = [to_pt(item, use_cuda) for item in word_indices]  # convert np.arrays to torch.tensors
    word_indices = [item.unsqueeze(-1) for item in word_indices]     # list of 5 tensors, each w size (batch,1)
    # return word_qvalues, word_indices
    return word_indices

# epsilon greedy choice: per-batch
# select either the specified maxq command phrase vs the specified random command phrase
# inputs: lists of tensors, one per word-position in output phrase, w/size(batch_len, 1)
# returns: list of tensors, one per word-position in output phrase, w/size(batch_len, 1)
def choose_epsilon_greedy(word_indices_maxq, word_indices_random, epsilon, use_cuda=True):
    # random number for epsilon greedy
    assert len(word_indices_maxq) == len(word_indices_random)  # lists (len = n_words) of (batch_size, 1)
    _batch_size = word_indices_maxq[0].size(0)
    rand_num = np.random.uniform(low=0.0, high=1.0, size=(_batch_size, 1)) # independtly random for each batch
    less_than_epsilon = (rand_num < epsilon).astype("float32")  # batch
    # note: one random number controls all n_words (=5) words in the p
    greater_than_epsilon = 1.0 - less_than_epsilon
    less_than_epsilon = to_pt(less_than_epsilon, use_cuda, type='float')
    greater_than_epsilon = to_pt(greater_than_epsilon, use_cuda, type='float')
    less_than_epsilon, greater_than_epsilon = less_than_epsilon.long(), greater_than_epsilon.long()
    # choose a word for each position in the output phrase
    chosen_indices = [  # per batch: choose either all 5 maxq words or all 5 random words
        less_than_epsilon * idx_random + greater_than_epsilon * idx_maxq
        for idx_random, idx_maxq in zip(word_indices_random, word_indices_maxq)
    ]
    return chosen_indices


def tally_word_qvalues(word_ranks, word_masks_np):
    """
    Arguments:
        word_ranks: Q values for each word by model.action_scorer.
        word_masks_np: Vocabulary masks for words depending on their type (verb, adj, noun).
    """
    # NOTE: compare simpler? impl when word_indices=chosen_indices are known (from replay buffer)
    #         chosen_indices = list(list(zip(*batch.word_indices)))
    #         chosen_indices = [torch.stack(item, 0) for item in chosen_indices]  # list (len n_words) of tensors size =(batch x 1)
    #   word_qvalues = [w_rank.gather(1, idx).squeeze(-1) for w_rank, idx in zip(word_ranks, chosen_indices)]  # list of batch
    #
    word_ranks_np = [to_np(item) for item in word_ranks]  # list (len n_words) of arrays (batch_len x n_vocab)
    word_ranks_np = [r - np.min(r) for r in word_ranks_np]  # minus the min value, so that all values are non-negative
    word_ranks_np = [r * m for r, m in zip(word_ranks_np, word_masks_np)]  # list (len n_words) of (batch x n_vocab)

    word_indices = [np.argmax(item, -1) for item in word_ranks_np]  # list (len=n_words) of np. (len=batch)
    # word_indices[w][b] -> index into vocab for output word position w of batch entry b

    word_qvalues = [[] for _ in word_masks_np]  # list of lists (one per output word position, each will be w/len=batch)

    batch_size = word_ranks[0].size(0)
    # print("batch_size=", batch_size, "len(word_qvalues)=", len(word_qvalues)) #batch_size=[16 or 24], len(word_qvalues)=5.
    for b in range(batch_size):
        for w in range(len(word_qvalues)):  # for each output word position
            word_qvalues[w].append(word_ranks[w][b][word_indices[w][b]])   # put the word rank for the chosen (maxQ) word
    word_qvalues = [torch.stack(item) for item in word_qvalues]     # apply torch.stack to each entry (each is a list of scalar tensors)
    return word_qvalues  # list (len=n_words) of tensors w size=(batch)  # 1 dim, since stack([s1,s2,s3])->>tensor([1,2,3])

def compute_per_step_rewards(scores, dones):
    """
    Compute rewards by agent. Note this is different from what the training/evaluation
    scripts do. Agent keeps track of scores and other game information for training purpose.

    """
    # mask = 1 if game is not finished or just finished at current step
    if not dones or len(dones) == 1:
        # it's not possible to finish a game at 0th step
        mask = [1.0 for _ in dones[-1]]
    else:
        assert len(dones) > 1
        mask = [1.0 if not dones[-2][i] else 0.0 for i in range(len(dones[-1]))]
    mask_np = np.array(mask, dtype='float32')
    # rewards returned by game engine are always the accumulated value received during the entire episode.
    # so the reward it gets in the current game step is the new value minus value at previous step.
    assert scores
    assert len(scores)
    rewards_np = np.array(scores[-1], dtype='float32')  # batch
    if len(scores) > 1:
        prev_rewards = np.array(scores[-2], dtype='float32')
        rewards_np = rewards_np - prev_rewards
    return rewards_np, mask_np

# def choose_command(word_ranks, word_masks_np, use_cuda, epsilon=0.0):
#     batch_size = word_ranks[0].size(0)
#     word_qvalues, word_indices_maxq = _choose_maxQ_command(word_ranks, word_masks_np, use_cuda)
#     if epsilon > 0.0:
#         _, word_indices_random = _choose_random_command(word_ranks, word_masks_np, use_cuda)
#         # random number for epsilon greedy
#         rand_num = np.random.uniform(low=0.0, high=1.0, size=(batch_size, 1))
#         less_than_epsilon = (rand_num < epsilon).astype("float32")  # batch
#         greater_than_epsilon = 1.0 - less_than_epsilon
#         less_than_epsilon = to_pt(less_than_epsilon, use_cuda, type='float')
#         greater_than_epsilon = to_pt(greater_than_epsilon, use_cuda, type='float')
#         less_than_epsilon, greater_than_epsilon = less_than_epsilon.long(), greater_than_epsilon.long()
#         chosen_indices = [
#             less_than_epsilon * idx_random + greater_than_epsilon * idx_maxq for idx_random, idx_maxq in
#             zip(word_indices_random, word_indices_maxq)]
#     else:
#         chosen_indices = word_indices_maxq
#     chosen_indices = [item.detach() for item in chosen_indices]
#     return word_qvalues, chosen_indices

# class CustomAgent:
#     """ Template agent for the TextWorld competition. """
#
#     # NOTE: MODIFIED to debug multiple inheritance
#     # def __init__(self) -> None:
#     def __init__(self, **kwargs) -> None:
#         print(f"CustomAgent.__init__ {kwargs}")
#         # super().__init__(**kwargs)
#         self._initialized = False
#         self._episode_has_started = False
#
#     def train(self) -> None:
#         """ Tell the agent it is in training mode. """
#         pass  # [You can insert code here.]
#
#     def eval(self) -> None:
#         """ Tell the agent it is in evaluation mode. """
#         pass  # [You can insert code here.]
#
#     def select_additional_infos(self) -> EnvInfos:
#         """
#         Returns what additional information should be made available at each game step.
#
#         Requested information will be included within the `infos` dictionary
#         passed to `CustomAgent.act()`. To request specific information, create a
#         :py:class:`textworld.EnvInfos <textworld.envs.wrappers.filter.EnvInfos>`
#         and set the appropriate attributes to `True`. The possible choices are:
#
#         * `description`: text description of the current room, i.e. output of the `look` command;
#         * `inventory`: text listing of the player's inventory, i.e. output of the `inventory` command;
#         * `max_score`: maximum reachable score of the game;
#         * `objective`: objective of the game described in text;
#         * `entities`: names of all entities in the game;
#         * `verbs`: verbs understood by the the game;
#         * `command_templates`: templates for commands understood by the the game;
#         * `admissible_commands`: all commands relevant to the current state;
#
#         In addition to the standard information, game specific information
#         can be requested by appending corresponding strings to the `extras`
#         attribute. For this competition, the possible extras are:
#
#         * `'recipe'`: description of the cookbook;
#         * `'walkthrough'`: one possible solution to the game (not guaranteed to be optimal);
#
#         Example:
#             Here is an example of how to request information and retrieve it.
#
#             >>> from textworld import EnvInfos
#             >>> request_infos = EnvInfos(description=True, inventory=True, extras=["recipe"])
#             ...
#             >>> env = gym.make(env_id)
#             >>> ob, infos = env.reset()
#             >>> print(infos["description"])
#             >>> print(infos["inventory"])
#             >>> print(infos["extra.recipe"])
#
#         Notes:
#             The following information *won't* be available at test time:
#
#             * 'walkthrough'
#
#             Requesting additional infos comes with some penalty (called handicap).
#             The exact penalty values will be defined in function of the average
#             scores achieved by agents using the same handicap.
#
#             Handicap is defined as follows
#                 max_score, has_won, has_lost,               # Handicap 0
#                 description, inventory, verbs, objective,   # Handicap 1
#                 command_templates,                          # Handicap 2
#                 entities,                                   # Handicap 3
#                 extras=["recipe"],                          # Handicap 4
#                 admissible_commands,                        # Handicap 5
#         """
#         return EnvInfos()
#
#     def _init(self) -> None:
#         """ Initialize the agent. """
#         self._initialized = True
#
#         # [You can insert code here.]
#
#     def _start_episode(self, obs: List[str], infos: Dict[str, List[Any]]) -> None:
#         """
#         Prepare the agent for the upcoming episode.
#
#         Arguments:
#             obs: Initial feedback for each game.
#             infos: Additional information for each game.
#         """
#         if not self._initialized:
#             self._init()
#
#         self._episode_has_started = True
#
#         # [You can insert code here.]
#
#     def _end_episode(self, obs: List[str], scores: List[int], infos: Dict[str, List[Any]]) -> None:
#         """
#         Tell the agent the episode has terminated.
#
#         Arguments:
#             obs: Previous command's feedback for each game.
#             score: The score obtained so far for each game.
#             infos: Additional information for each game.
#         """
#         self._episode_has_started = False
#
#         # [You can insert code here.]
#
#     def act(self, obs: List[str], scores: List[int], dones: List[bool], infos: Dict[str, List[Any]]) -> Optional[List[str]]:
#         """
#         Acts upon the current list of observations.
#
#         One text command must be returned for each observation.
#
#         Arguments:
#             obs: Previous command's feedback for each game.
#             scores: The score obtained so far for each game.
#             dones: Whether a game is finished.
#             infos: Additional information for each game.
#
#         Returns:
#             Text commands to be performed (one per observation).
#             If episode had ended (e.g. `all(dones)`), the returned
#             value is ignored.
#
#         Notes:
#             Commands returned for games marked as `done` have no effect.
#             The states for finished games are simply copy over until all
#             games are done.
#         """
#         if all(dones):
#             self._end_episode(obs, scores, infos)
#             return  # Nothing to return.
#         elif not self._episode_has_started:
#             self._start_episode(obs, infos)
#
#         # [Insert your code here to obtain the commands.]
#         return ["wait"] * len(obs)  # No-op


class AgentDQN(pl.LightningModule):
    MODE_TRAIN = 'train'
    MODE_EVAL = 'eval'

    def __init__(self, cfg, **kwargs):
        print(f"AgentDQN.__init__ {cfg} {kwargs}")
        # CustomAgent.__init__(self)
        seedval = cfg.general.random_seed

        self.gym_env = None
        self.mode = self.MODE_TRAIN

        # pl.LightningModule.__init__(self, **kwargs)
        super().__init__(**kwargs)

        self.hparams = cfg   # will be saved in checkpoints by PyTorchLightning
        self.vocab = WordVocab(vocab_file=cfg.general.vocab_words)

        # NOTE: base_vocab is an optimization hack that won't quite work with async parallel batches
        self.qgym = QaitGym(random_seed=seedval)  #, base_vocab=self.vocab)  # it also doesn't seem to actually speed things up

        # training
        # self.batch_size = cfg.training.batch_size
        self.max_nb_steps_per_episode = cfg.training.max_nb_steps_per_episode
        self.sync_rate = cfg.training.sync_rate
        # self.nb_epochs = cfg.training.nb_epochs

        # Set the random seed manually for reproducibility.
        np.random.seed(seedval)
        torch.manual_seed(seedval)
        if torch.cuda.is_available():
            if not cfg.general.use_cuda:
                print("WARNING: CUDA device detected but 'use_cuda: false' found in config.yaml")
                self.use_cuda = False
            else:
                torch.backends.cudnn.deterministic = True
                torch.cuda.manual_seed(seedval)
                self.use_cuda = True
        else:
            self.use_cuda = False

        self.save_frequency = cfg.checkpoint.save_frequency

        self.discount_gamma = cfg.general.discount_gamma
        self.replay_batch_size = cfg.general.replay_batch_size

        # epsilon greedy
        self.epsilon_anneal_episodes = cfg.general.epsilon_anneal_episodes
        self.epsilon_anneal_from = cfg.general.epsilon_anneal_from
        self.epsilon_anneal_to = cfg.general.epsilon_anneal_to
        self.epsilon = self.epsilon_anneal_from
        self.update_per_k_game_steps = cfg.general.update_per_k_game_steps

        self.current_episode = 0
        self.current_step = 0
        self.history_avg_scores = HistoryScoreCache(capacity=1000)
        self.best_avg_score_so_far = 0.0

        self.experiment_tag = cfg.checkpoint.experiment_tag
        self.model_checkpoint_path = cfg.checkpoint.model_checkpoint_path

        self.clip_grad_norm = cfg.training.optimizer.clip_grad_norm

        self.model = LSTM_DQN(model_config=cfg.model,
                              word_vocab=self.vocab.word_vocab,
                              # DEFAULT: generate_length=5,  # how many output words to generate
                              # generate_length=len(self.vocab.word_masks_np), # but vocab.word_masks_np get initialized later
                              enable_cuda=self.use_cuda)
        self.target_net = LSTM_DQN(model_config=cfg.model,
                              word_vocab=self.vocab.word_vocab,
                              # DEFAULT: generate_length=5,  # how many output words to generate
                              # generate_length=len(self.vocab.word_masks_np), # but vocab.word_masks_np get initialized later
                              enable_cuda=self.use_cuda)
        if cfg.checkpoint.load_pretrained:
            self.load_pretrained_model(
                self.model_checkpoint_path + '/' + cfg.checkpoint.pretrained_experiment_tag + '.pt')
        if self.use_cuda:
            self.model.cuda()

        self.replay_memory = PrioritizedReplayMemory(
                                cfg.general.replay_memory_capacity,
                                priority_fraction=cfg.general.replay_memory_priority_fraction)
        # optimizer
        self.learning_rate = cfg.training.optimizer.learning_rate
        self.optimizer = self.configure_optimizers()[0]

    def set_mode(self, mode):
        assert mode == self.MODE_TRAIN or mode == self.MODE_EVAL, str(mode)
        self.mode = mode

    def is_eval_mode(self) -> bool:
        assert self.mode == self.MODE_TRAIN or self.mode == self.MODE_EVAL, self.mode
        return self.mode == self.MODE_EVAL

    def train(self, mode=True):
        """
        Tell the agent that it's training phase.
        """
        self.model.train(mode)
        if mode:
            self.set_mode(self.MODE_TRAIN)
        else:
            self.set_mode(self.MODE_EVAL)
        super().train(mode)

    def eval(self):
        """
        Tell the agent that it's evaluation phase.
        """
        self.model.eval()
        self.set_mode(self.MODE_EVAL)
        super().eval()

    def prepopulate_replay_buffer(self, steps=1000):

        """
        Carries out several random steps through the environment to initially fill
        up the replay buffer with experiences

        Args:
            steps: number of random steps to populate the buffer with
        """
        print(f"\nwarm_start_steps:{steps}")   # "obs_size:{obs_size}")
        for i in range(steps):
            _obs_, scores, dones, _infos_ = self.env_experience(self._prev_obs, self.scores[-1], self.dones[-1], self._prev_infos)

    def step_env(self, commands: List[str], dones: List[bool]):  #-> List[str], List[int], List[bool], List[Map]
        # steps = [step + int(not done) for step, done in zip(steps, dones)]  # Increase step counts.
        for idx, done in enumerate(dones):
            if not done:
                self.num_steps[idx] += 1
            obs, scores, dones, infos = self.gym_env.step(commands)
            return obs, scores, dones, infos


    def env_experience(self, obs, scores, dones, infos):
        __commands__, act_idlist, obs_idlist = self.select_next_action(obs, scores, dones, infos)
        commands = self.vocab.get_chosen_strings(act_idlist, strip_padding=True)
        assert len(__commands__) == len(commands), f"{__commands__} {commands}"
        for i, c in enumerate(__commands__):
            while ' the ' in c:
                c = c.replace(' the ', ' ')
            assert c.lower() == commands[i], f"{c} should== {commands[i]}"

        obs, rewards, dones, infos = self.step_env(commands, dones)
        accum_score = [score+reward for score, reward in zip(scores, rewards)]

        self._cache_transitions = self.observe_action_results(rewards, dones, obs, obs_idlist, act_idlist)
        self._prev_infos = infos  # HACK: save for use in next call to training_step()
        # HACK: observe_action_results =>
        #    self.scores[-1] = scores; self.dones[-1] = dones;
        #    self._prev_obs = obs
        #    self.cache_description_id_list = obs_idlist  # token ids for prev obs
        #    self.cache_chosen_indices = act_idlist  # token ids for prev action
        #    self.step_reward = rewards_np from compute_per_step_rewards()

        return obs, accum_score, dones, infos

    def observe_action_results(self, rewards, dones, obs, obs_idlist, act_idlist):
        # append scores / dones from previous step into memory

        # self.scores.append(scores)
        if len(self.scores) > 0:
            prev_scores = self.scores[-1]
        else:
            prev_scores = [0] * len(rewards)
        new_scores = [prev_score + reward for prev_score, reward in zip(prev_scores, rewards)]
        self.scores.append(new_scores)
        self.dones.append(dones)
        rewards_np, mask_np = compute_per_step_rewards(self.scores, self.dones)
        self.step_reward = rewards_np

        transitions = self.get_transitions_for_replay(rewards_np, mask_np, obs_idlist)  #, act_idlist)
        if not self.is_eval_mode() and transitions:
            for is_priority, transition in transitions:
                self.replay_memory.push(is_priority=is_priority, transition=transition)

        # cache info from current game step for constructing next Transition
        self._prev_obs = obs  # save for constructing transition during next step
        self.cache_description_id_list = obs_idlist  # token ids for prev obs
        self.cache_chosen_indices = act_idlist  # token ids for prev action

        if all(dones):
            self.qait_env._on_episode_end()  # log/display some stats
        return transitions

    def get_transitions_for_replay(self, rewards_np, mask_np, obs_idlist):  #, act_idlist):
        if not self.cache_chosen_indices:  # skip if this is the first step (no transition is yet available
            return None
        transitions = []
        mask_pt = to_pt(mask_np, self.use_cuda, type='float')
        rewards_pt = to_pt(rewards_np, self.use_cuda, type='float')
        # push info from previous game step into replay memory
        # if self.current_step > 0:
        for b in range(len(mask_np)):   # for each in batch
            if mask_np[b] == 0:
                continue
            is_priority = rewards_np[b] > 0.0

            transition = Transition(
                obs_word_ids=self.cache_description_id_list[b],
                cmd_word_ids=[item[b] for item in self.cache_chosen_indices],
                reward=rewards_pt[b],
                mask=mask_pt[b],
                done=self.dones[-1][b],
                next_obs_word_ids=obs_idlist[b],
                next_word_masks=[word_mask[b]
                                 for word_mask in self.vocab.word_masks_np])
            transitions.append((is_priority, transition))
        return transitions

    def run_episode(self, gamefiles: List[str]) -> Tuple[List[int], List[int]]:
        """ returns two lists (each containing one value per game in batch): final_score, number_of_steps"""
        # batch_size = self.batch_size
        # assert len(gamefiles) == batch_size, f"{batch_size} {len(gamefiles)}"

        obs, infos = self.initialize_episode(gamefiles)
        # self.num_steps = [0] * len(obs)   # step counts, local var (not really used, except for logging/print out at end)
        # self.initial_observation(obs, infos)

        scores = [0] * len(obs)
        dones = [False] * len(obs)
        while not all(dones):

            obs, scores, dones, infos = self.env_experience(obs, scores, dones, infos)
            batch = self.get_next_training_batch()
            if batch and not self.is_eval_mode():
                if self.current_step > 0 and self.current_step % self.update_per_k_game_steps == 0:
                    loss = self.calculate_batch_loss(batch)
                    if loss is not None:
                        # Backpropagate
                        self.backpropagate(loss)
            else:
                print("WARNING: **** no batch => no loss ****")

        # Let the agent know the game is done and see the final observation
        # self.select_next_action(obs, scores, dones, infos)
        self._compute_episode_rewards()
        self._maybe_save_checkpoint()
        self.current_episode += 1
        self._anneal_epsilon()
        return scores, self.num_steps   #, steps

    def save_checkpoint(self, episode_num):
        save_to = self.model_checkpoint_path + '/' + self.experiment_tag + "_episode_" + str(episode_num) + ".pt"
        if not os.path.isdir(self.model_checkpoint_path):
            os.mkdir(self.model_checkpoint_path)
        torch.save(self.model.state_dict(), save_to)
        print("========= saved checkpoint =========")

    def load_pretrained_model(self, load_from):
        """
        Load pretrained checkpoint from file.

        Arguments:
            load_from: File name of the pretrained model checkpoint.
        """
        # print("loading model from %s\n" % (load_from))
        try:
            if self.use_cuda:
                state_dict = torch.load(load_from)
            else:
                state_dict = torch.load(load_from, map_location='cpu')
            self.model.load_state_dict(state_dict)
        except:
            print("Failed to load checkpoint...")

    def get_next_training_batch(self):
        """
        Update neural model in agent. In this example we follow algorithm
        of updating model in dqn with replay memory.

        """
        if len(self.replay_memory) < self.replay_batch_size:
            return None
        transitions = self.replay_memory.sample(self.replay_batch_size)
        batch = Transition(*zip(*transitions))
        return batch

    def calculate_batch_loss(self, batch):

        observation_id_list = pad_sequences(batch.obs_word_ids, maxlen=max_len(batch.obs_word_ids)).astype('int32')
        input_observation = to_pt(observation_id_list, self.use_cuda)
        next_observation_id_list = pad_sequences(batch.next_obs_word_ids, maxlen=max_len(batch.next_obs_word_ids)).astype('int32')
        next_input_observation = to_pt(next_observation_id_list, self.use_cuda)
        chosen_indices = list(list(zip(*batch.cmd_word_ids)))
        chosen_indices = [torch.stack(item, 0) for item in chosen_indices]  # list (len n_words) of tensors size =(batch x 1)

        word_ranks = self.model.infer_word_ranks(input_observation)  # list (len = n_words) of (batch x vocab) (one row per potential output word)
        word_qvalues = [w_rank.gather(1, idx).squeeze(-1) for w_rank, idx in zip(word_ranks, chosen_indices)]  # list of batch
        q_value = torch.mean(torch.stack(word_qvalues, -1), -1)  # batch

        next_word_ranks = self.model.infer_word_ranks(next_input_observation)  # batch x n_verb, batch x n_noun, batch x n_second_noun
        next_word_masks = list(list(zip(*batch.next_word_masks)))
        next_word_masks = [np.stack(item, 0) for item in next_word_masks]
        next_word_qvalues = tally_word_qvalues(next_word_ranks, next_word_masks)  # batch

        next_q_value = torch.mean(torch.stack(next_word_qvalues, -1), -1)  # batch
        next_q_value = next_q_value.detach()  # make a copy, detatched from autograd graph (don't backprop)

        rewards = torch.stack(batch.reward)  # batch
        not_done = 1.0 - np.array(batch.done, dtype='float32')  # batch
        not_done = to_pt(not_done, self.use_cuda, type='float')
        rewards = rewards + not_done * next_q_value * self.discount_gamma  # batch

        mask = torch.stack(batch.mask)  # batch
        loss = F.smooth_l1_loss(q_value * mask, rewards * mask)
        return loss

    def backpropagate(self, loss):
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
        self.optimizer.step()  # apply gradients

    def _compute_episode_rewards(self):
        dones = []
        for d in self.dones:
            d = np.array([float(dd) for dd in d], dtype='float32')
            dones.append(d)
        dones = np.array(dones)
        step_used = 1.0 - dones
        self.step_used_before_done = np.sum(step_used, 0)  # batch

        if len(self.scores):
            self.final_rewards = np.array(self.scores[-1], dtype='float32')  # batch
            self.history_avg_scores.push(np.mean(self.final_rewards))
        else:
            print(f"!!! WARNING: finish() called but self.scores={self.scores}")

    def _maybe_save_checkpoint(self):
        # save checkpoint
        if not self.is_eval_mode() and self.current_episode % self.save_frequency == 0:
            avg_score = self.history_avg_scores.get_avg()
            if avg_score > self.best_avg_score_so_far:
                self.best_avg_score_so_far = avg_score

                self.save_checkpoint(self.current_episode)

    def _anneal_epsilon(self):
        # self.current_episode += 1
        # annealing
        if self.current_episode < self.epsilon_anneal_episodes:
            self.epsilon -= (self.epsilon_anneal_from - self.epsilon_anneal_to) / float(self.epsilon_anneal_episodes)

    def training_step(self, batch, batch_idx) -> Dict[str, Any]:
        """
        Carries out a single step through the environment to update the replay buffer.
        Then calculates loss based on the minibatch received

        Args:
            batch: current mini batch of replay data (Tuple[torch.Tensor, torch.Tensor, ...])
            batch_idx: batch number

        Returns:
            Training loss and log metrics
        """
        # device = self.get_device(batch)
        # epsilon = max(self.eps_end, self.eps_start -
        #               self.global_step + 1 / self.eps_last_frame)
        #
        # # step through environment with agent
        # reward, done = self.agent.play_step(self.net, epsilon, device)
        _obs_, reward, dones, _infos_ = self.env_experience(self._prev_obs, self.scores[-1], self.dones[-1], self._prev_infos)

        # HACK:
        # reward = self.step_reward
        self.episode_reward = np.array(self.scores[-1], dtype='float32')

        #
        # # calculates training loss
        if not self._cache_transitions:
            loss = None
        else:
            batch_transitions = [transition for _, transition in self._cache_transitions]
            batch = Transition(*zip(*batch_transitions))
            loss = self.calculate_batch_loss(batch)
        #
        # if all(dones):
        #     self.total_reward = self.episode_reward
        #     self.episode_reward = np.array([0.0]*len(_obs_))

        # Soft update of target network
        if self.global_step % self.sync_rate == 0:
            self.target_net.load_state_dict(self.model.state_dict())
        #
        # # log = {'total_reward': torch.tensor(self.total_reward).to(device),
        # #        'reward': torch.tensor(reward).to(device),
        # #        'steps': torch.tensor(self.global_step).to(device)}
        self.log('reward', torch.tensor(reward, dtype=torch.float32), on_step=True, on_epoch=True)
        self.log('episode_reward', torch.tensor(self.episode_reward, dtype=torch.float32), on_step=True, on_epoch=True)
        self.log('steps', torch.tensor(self.global_step, dtype=torch.float32), on_epoch=True)
        if loss:
            self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        if all(dones):
            #UGLY HACK TODO: (very soon) get rid of this!!!!
            self.prepare_for_fake_replay()
        return loss   # loss is a torch.Tensor

    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer"""
        # optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = torch.optim.Adam(parameters, lr=self.learning_rate)
        return [optimizer]

    # def __dataloader(self) -> DataLoader:
    #     """Initialize the Replay Buffer dataset used for retrieving experiences"""
    #     dataset = RLDataset(self.buffer, self.episode_length)
    #     dataloader = DataLoader(
    #         dataset=dataset,
    #         batch_size=self.batch_size,
    #         sampler=None,
    #     )
    #     return dataloader
    #
    # def train_dataloader(self) -> DataLoader:
    #     """Get train loader"""
    #     return self.__dataloader()

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch"""
        return batch[0].device.index if self.on_gpu else 'cpu'


class FtwcAgent(AgentDQN):
    def __init__(self, cfg: Dict[str, Any], **kwargs):
        """
        Arguments:
            vocab: words supported.
        """
        super().__init__(cfg, **kwargs)
        # self._episode_initialized = False
        self.requested_infos = self.select_additional_infos()

    def select_additional_infos(self) -> EnvInfos:
        """
        Returns what additional information should be made available at each game step.

        Requested information will be included within the `infos` dictionary
        passed to `CustomAgent.act()`. To request specific information, create a
        :py:class:`textworld.EnvInfos <textworld.envs.wrappers.filter.EnvInfos>`
        and set the appropriate attributes to `True`. The possible choices are:

        * `description`: text description of the current room, i.e. output of the `look` command;
        * `inventory`: text listing of the player's inventory, i.e. output of the `inventory` command;
        * `max_score`: maximum reachable score of the game;
        * `objective`: objective of the game described in text;
        * `entities`: names of all entities in the game;
        * `verbs`: verbs understood by the the game;
        * `command_templates`: templates for commands understood by the the game;
        * `admissible_commands`: all commands relevant to the current state;

        In addition to the standard information, game specific information
        can be requested by appending corresponding strings to the `extras`
        attribute. For this competition, the possible extras are:

        * `'recipe'`: description of the cookbook;
        * `'walkthrough'`: one possible solution to the game (not guaranteed to be optimal);

        Example:
            Here is an example of how to request information and retrieve it.

            >>> from textworld import EnvInfos
            >>> request_infos = EnvInfos(description=True, inventory=True, extras=["recipe"])
            ...
            >>> env = gym.make(env_id)
            >>> ob, infos = env.reset()
            >>> print(infos["description"])
            >>> print(infos["inventory"])
            >>> print(infos["extra.recipe"])

        Notes:
            The following information *won't* be available at test time:

            * 'walkthrough', 'facts'
        """
        _REQUESTED_EXTRAS = ["recipe", "uuid"]

        request_infos = EnvInfos(
            description=True,
            inventory=True,
            location=True,
            entities=True,
            verbs=True,
            facts=True,   # use ground truth facts about the world (since this is a training oracle)
            extras=_REQUESTED_EXTRAS
        )
        return request_infos


    def initialize_episode(self, gamefiles=None):
        """
        Prepare the agent for the upcoming games.

        Arguments:
            obs: Previous command's feedback for each game.
            infos: Additional information for each game.
        """
        if not gamefiles:
            assert False, "Missing arg: gamefiles"
        self._gamefiles = gamefiles   #HACK
        batch_size = len(gamefiles)
        self.qait_env = self.qgym.make_batch_env(gamefiles, self.vocab,
                                            request_infos=self.requested_infos,
                                            batch_size=len(gamefiles),
                                            max_episode_steps=self.max_nb_steps_per_episode)  #self.cfg.training.batch_size)

        wrapped_env = ScoreToRewardWrapper(self.qait_env)
        self.gym_env = wrapped_env
        obs, infos = self.gym_env.reset()
        self.vocab.init_from_infos_lists(infos['verbs'], infos['entities'])

        # self.start_episode_infos(obs, infos)
        assert len(self.vocab.word_masks_np) == self.model.generate_length, \
                f"{len(self.vocab.word_masks_np)} SHOULD == {self.model.generate_length}"  # == 5

        batch_size = len(obs)
        self.prev_actions = ['' for _ in range(batch_size)]
        self._prev_obs = obs
        self._prev_infos = infos
        self.cache_description_id_list = None   # numerical version of .prev_obs
        self.cache_chosen_indices = None        # numerical version of .prev_actions
        self.current_step = 0
        # reset agent, get vocabulary masks for verbs / adjectives / nouns
        self.scores = []
        self.num_steps = [0] * batch_size
        self.dones = []
        self.episode_reward = np.array([0.0]* batch_size, dtype='float32')

        return obs, infos

    def prepare_for_fake_replay(self):
        self.initialize_episode(self._gamefiles)
        self.scores.append([0])
        self.dones.append([False])
        self.prepopulate_replay_buffer(steps=2)

    def prepare_input_for_action_selection(self, obs: List[str], infos: Dict[str, List[Any]]):
        """
        Get all the available information, and concat them together to be tensor for
        a neural model. we use post padding here, all information are tokenized here.

        Arguments:
            obs: Previous command's feedback for each game.
            infos: Additional information for each game.
        """
        # word2id = self.vocab.word2id
        inventory_id_list = self.vocab.token_id_lists_from_strings(infos["inventory"])
        feedback_id_list = self.vocab.token_id_lists_from_strings(obs, str_type="feedback")
        quest_id_list = self.vocab.token_id_lists_from_strings(infos["extra.recipe"])
        prev_action_id_list = self.vocab.token_id_lists_from_strings(self.prev_actions)
        description_id_list = self.vocab.token_id_lists_from_strings(infos["description"], subst_if_empty=['end'])
        input_token_ids = [_d + _i + _q + _f + _pa for (_d, _i, _q, _f, _pa) in zip(description_id_list,
                                                                                        inventory_id_list,
                                                                                        quest_id_list,
                                                                                        feedback_id_list,
                                                                                        prev_action_id_list)]

        input_token_ids = pad_sequences(input_token_ids, maxlen=max_len(input_token_ids)).astype('int32')
        input_tensor = to_pt(input_token_ids, self.use_cuda)
        return input_tensor, input_token_ids

    def select_next_action(self, obs: List[str], scores: List[int], dones: List[bool], infos: Dict[str, List[Any]]) -> List[str]:
        """
        Acts upon the current list of observations.

        One text command must be returned for each observation.

        Arguments:
            obs: Previous command's feedback for each game.
            score: The score obtained so far for each game (at previous step).
            done: Whether a game is finished (at previous step).
            infos: Additional information for each game.

        Returns:
            Text commands to be performed (one per observation).

        Notes:
            Commands returned for games marked as `done` have no effect.
            The states for finished games are simply copy over until all
            games are done, in which case `CustomAgent.finish()` is called
            instead.
        """
        # if not self._episode_has_started:
        #     self._start_episode(obs, infos)

        # if self.is_eval_mode():
        #     return self.select_next_action_eval(obs, scores, dones, infos)
        #
        # if not self.is_eval_mode():
        #     # assert self.mode == MODE_TRAIN
        #     # compute previous step's rewards and masks
        #     rewards_pt, mask_pt = self.compute_reward(self.scores, self.dones)

        input_tensor, __input_token_ids__ = self.prepare_input_for_action_selection(obs, infos)

        word_ranks = self.model.infer_word_ranks(input_tensor)  # list of batch x vocab
        assert word_ranks[0].size(0) == input_tensor.size(0)   # refactoring

        # generate commands for one game step, epsilon greedy is applied, i.e.,
        # there is epsilon of chance to generate random commands
        # _, chosen_indices = choose_command(word_ranks,
        #                                    self.vocab.word_masks_np,
        #                                    self.use_cuda,
        #                                    epsilon=(0.0 if self.is_eval_mode() else self.epsilon))

        word_indices_maxq = _choose_maxQ_command(word_ranks, self.vocab.word_masks_np, self.use_cuda)
        if self.is_eval_mode() or self.epsilon <= 0.0:
            chosen_indices = word_indices_maxq
        else:  # if not self.is_eval_mode() and self.epsilon > 0.0:
            word_indices_random = _choose_random_command(word_ranks, self.vocab.word_masks_np, self.use_cuda)
            chosen_indices = choose_epsilon_greedy(word_indices_maxq, word_indices_random, self.epsilon)

        chosen_indices = [item.detach() for item in chosen_indices]
        use_oracle = True or self.is_eval_mode()
        if use_oracle:
            chosen_strings = []
            # for idx, (obstxt, agent) in enumerate(zip(obs, self.qgym.tw_oracles)):
            for idx in range(len(obs)):
                # verbose = (self.current_step == 0)
                # actiontxt = self.qgym.invoke_oracle(idx, obstxt, infos, verbose=verbose)
                actiontxt = infos['tw_o_step'][idx]
                chosen_strings.append(actiontxt)

            #TODO: if not self.is_eval_mode() compute appropriate chosen_indices
            oracle_indices = self.vocab.command_strings_to_np(chosen_strings)  # pad and convert to token ids

            chosen_indices0 = chosen_indices
            chosen_indices = [to_pt(item, self.use_cuda) for item in oracle_indices]
            chosen_indices = [item.unsqueeze(-1) for item in chosen_indices]  # list of 5 tensors, each w size: batch x 1
            print(f"ORACLE cmds: {chosen_strings} -> {oracle_indices} -> {self.vocab.get_chosen_strings(chosen_indices, strip_padding=True)}")
            # print(f'DEBUGGING use_oracle: len:{len(chosen_indices)}=={len(chosen_indices0)} shape: {chosen_indices[0].shape}=={chosen_indices0[0].shape} {chosen_indices} {chosen_indices0}')
            assert len(chosen_indices) == len(chosen_indices0)
            for _i in range(len(chosen_indices0)):
                assert chosen_indices0[_i].shape == chosen_indices[_i].shape

        # else:
        chosen_strings = self.vocab.get_chosen_strings(chosen_indices)
        self.prev_actions = chosen_strings

        self.current_step += 1
        return self.prev_actions, chosen_indices, __input_token_ids__

    def validation_step(self, batch, batch_idx):
        print(f"\n=========== VALIDATION_STEP [{batch_idx}] {batch}\n")
        return self._shared_eval(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        print(f"\n=========== TEST_STEP [{batch_idx}] {batch}\n")
        return self._shared_eval(batch, batch_idx, 'test')

    def _shared_eval(self, batch, batch_idx, prefix):
        scores, steps = self.run_episode(batch)
        # x, _ = batch
        # representation = self.encoder(x)
        # x_hat = self.decoder(representation)
        #
        # loss = self.metric(x, x_hat)
        print("EVAL results:", scores, steps)
        for score in scores:
            self.log(f"{prefix}_score", torch.tensor(score, dtype=torch.float16), on_step=True, on_epoch=True)
        for nsteps in steps:  # total steps, per env in batch
            self.log(f"{prefix}_nsteps", torch.tensor(nsteps, dtype=torch.int16), on_step=True, on_epoch=True)
        return None
