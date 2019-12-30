import os
import random
import yaml
import copy
from typing import List, Dict, Any
from collections import namedtuple

import spacy
import numpy as np

import torch
import torch.nn.functional as F

from textworld import EnvInfos

from model import LSTM_DQN
from generic import to_np, to_pt, preproc, _words_to_ids, get_token_ids_for_items, pad_sequences, max_len

from symbolic.nail import NailAgent
from symbolic.event import NeedToAcquire, NeedToGoTo
from symbolic.gv import dbg
from twutils.twlogic import filter_observables

# a snapshot of state to be stored in replay memory
Transition = namedtuple('Transition', ('observation_id_list', 'word_indices',
                                       'reward', 'mask', 'done',
                                       'next_observation_id_list',
                                       'next_word_masks'))


class HistoryScoreCache(object):

    def __init__(self, capacity=1):
        self.capacity = capacity
        self.reset()

    def push(self, stuff):
        """stuff is float."""
        if len(self.memory) < self.capacity:
            self.memory.append(stuff)
        else:
            self.memory = self.memory[1:] + [stuff]

    def get_avg(self):
        return np.mean(np.array(self.memory))

    def reset(self):
        self.memory = []

    def __len__(self):
        return len(self.memory)


class PrioritizedReplayMemory(object):

    def __init__(self, capacity=100000, priority_fraction=0.0):
        # prioritized replay memory
        self.priority_fraction = priority_fraction
        self.alpha_capacity = int(capacity * priority_fraction)
        self.beta_capacity = capacity - self.alpha_capacity
        self.alpha_memory, self.beta_memory = [], []
        self.alpha_position, self.beta_position = 0, 0

    def push(self, is_prior=False, transition:Transition=None):
        """Saves a transition."""
        assert transition is not None

        if self.priority_fraction == 0.0:
            is_prior = False
        if is_prior:
            if len(self.alpha_memory) < self.alpha_capacity:
                self.alpha_memory.append(None)
            self.alpha_memory[self.alpha_position] = transition #Transition(*args)
            self.alpha_position = (self.alpha_position + 1) % self.alpha_capacity
        else:
            if len(self.beta_memory) < self.beta_capacity:
                self.beta_memory.append(None)
            self.beta_memory[self.beta_position] = transition
            self.beta_position = (self.beta_position + 1) % self.beta_capacity

    def sample(self, batch_size):
        if self.priority_fraction == 0.0:
            from_beta = min(batch_size, len(self.beta_memory))
            res = random.sample(self.beta_memory, from_beta)
        else:
            from_alpha = min(int(self.priority_fraction * batch_size), len(self.alpha_memory))
            from_beta = min(batch_size - int(self.priority_fraction * batch_size), len(self.beta_memory))
            res = random.sample(self.alpha_memory, from_alpha) + random.sample(self.beta_memory, from_beta)
        random.shuffle(res)
        return res

    def __len__(self):
        return len(self.alpha_memory) + len(self.beta_memory)


class AgentDQN:
    def __init__(self, config, vocab, use_cuda):
        self.model_config = config['model']
        self.experiment_tag = config['checkpoint']['experiment_tag']
        self.model_checkpoint_path = config['checkpoint']['model_checkpoint_path']
        self.discount_gamma = config['general']['discount_gamma']
        self.clip_grad_norm = config['training']['optimizer']['clip_grad_norm']

        self.replay_batch_size = config['general']['replay_batch_size']

        self.use_cuda = use_cuda
        self.vocab = vocab
        self.model = LSTM_DQN(model_config=self.model_config,
                              word_vocab=self.vocab.word_vocab,
                              enable_cuda=self.use_cuda)
        if config['checkpoint']['load_pretrained']:
            self.load_pretrained_model(self.model_checkpoint_path + '/' +
                                       config['checkpoint']['pretrained_experiment_tag'] + '.pt')
        if self.use_cuda:
            self.model.cuda()

        self.replay_memory = PrioritizedReplayMemory(
                                config['general']['replay_memory_capacity'],
                                priority_fraction=config['general']['replay_memory_priority_fraction'])
        # optimizer
        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.Adam(parameters, lr=config['training']['optimizer']['learning_rate'])


    def infer_word_ranks(self, input_description):
        """
        Given input description tensor, call model forward, to get Q values of words.

        Arguments:
            input_description: Input tensors, which include all the information chosen in
            select_additional_infos() concatenated together.
        """
        state_representation = self.model.representation_generator(input_description)
        word_ranks = self.model.action_scorer(state_representation)  # each element in list has batch x n_vocab size
        return word_ranks

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

    def update(self):
        """
        Update neural model in agent. In this example we follow algorithm
        of updating model in dqn with replay memory.

        """
        if len(self.replay_memory) < self.replay_batch_size:
            return None
        transitions = self.replay_memory.sample(self.replay_batch_size)
        batch = Transition(*zip(*transitions))

        observation_id_list = pad_sequences(batch.observation_id_list, maxlen=max_len(batch.observation_id_list)).astype('int32')
        input_observation = to_pt(observation_id_list, self.use_cuda)
        next_observation_id_list = pad_sequences(batch.next_observation_id_list, maxlen=max_len(batch.next_observation_id_list)).astype('int32')
        next_input_observation = to_pt(next_observation_id_list, self.use_cuda)
        chosen_indices = list(list(zip(*batch.word_indices)))
        chosen_indices = [torch.stack(item, 0) for item in chosen_indices]  # list of batch x 1

        word_ranks = self.infer_word_ranks(input_observation)  # list of batch x vocab, len=5 (one per potential output word)
        word_qvalues = [w_rank.gather(1, idx).squeeze(-1) for w_rank, idx in zip(word_ranks, chosen_indices)]  # list of batch
        q_value = torch.mean(torch.stack(word_qvalues, -1), -1)  # batch

        next_word_ranks = self.infer_word_ranks(next_input_observation)  # batch x n_verb, batch x n_noun, batchx n_second_noun
        next_word_masks = list(list(zip(*batch.next_word_masks)))
        next_word_masks = [np.stack(item, 0) for item in next_word_masks]
        next_word_qvalues, _ = _choose_maxQ_command(next_word_ranks, next_word_masks, self.use_cuda)
        next_q_value = torch.mean(torch.stack(next_word_qvalues, -1), -1)  # batch
        next_q_value = next_q_value.detach()

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


def _choose_random_command(word_ranks, word_masks_np, use_cuda):
    """
    Generate a command randomly, for epsilon greedy.

    Arguments:
        word_ranks: Q values for each word by model.action_scorer.
        word_masks_np: Vocabulary masks for words depending on their type (verb, adj, noun, adj2, noun2).
    """

    batch_size = word_ranks[0].size(0)
    # print("batch_size=", batch_size, len(word_masks_np))
    assert len(word_ranks) == len(word_masks_np)

    word_ranks_np = [to_np(item) for item in word_ranks]  # list of (batch x n_vocab) arrays, len=5 (5 word output phrases)
    # word_ranks_np = [r - np.min(r) for r in word_ranks_np]  # minus the min value, so that all values are non-negative
    word_ranks_np = [r * m for r, m in zip(word_ranks_np, word_masks_np)]  # list of batch x n_vocab

    word_indices = []
    for i in range(len(word_ranks_np)):  # len=5 (verb, adj1, noun1, adj2, noun2)
        indices = []
        for j in range(batch_size):
            msk = word_masks_np[i][j]  # msk is of len = vocab, j is index into batch
            indices.append(np.random.choice(len(msk), p=msk / np.sum(msk, -1)))  # choose from non-zero entries of msk
        word_indices.append(np.array(indices))
    # word_indices: list of batch

    word_qvalues = [[] for _ in word_masks_np]
    for i in range(batch_size):
        for j in range(len(word_qvalues)):
            word_qvalues[j].append(word_ranks[j][i][word_indices[j][i]])
    word_qvalues = [torch.stack(item) for item in word_qvalues]
    word_indices = [to_pt(item, use_cuda) for item in word_indices]
    word_indices = [item.unsqueeze(-1) for item in word_indices]  # list of batch x 1
    return word_qvalues, word_indices


def _choose_maxQ_command(word_ranks, word_masks_np, use_cuda):
    """
    Generate a command by maximum q values, for epsilon greedy.

    Arguments:
        word_ranks: Q values for each word by model.action_scorer.
        word_masks_np: Vocabulary masks for words depending on their type (verb, adj, noun).
    """
    batch_size = word_ranks[0].size(0)
    word_ranks_np = [to_np(item) for item in word_ranks]  # list of arrays, batch_len x n_vocab
    word_ranks_np = [r - np.min(r) for r in word_ranks_np]  # minus the min value, so that all values are non-negative
    word_ranks_np = [r * m for r, m in zip(word_ranks_np, word_masks_np)]  # list of batch x n_vocab

    word_indices = [np.argmax(item, -1) for item in word_ranks_np]  # list of arrays of len = batch

    word_qvalues = [[] for _ in word_masks_np]
    # print("batch_size=", batch_size, "len(word_qvalues)=", len(word_qvalues)) #batch_size=[16 or 24], len(word_qvalues)=5.
    for i in range(batch_size):
        for j in range(len(word_qvalues)):
            word_qvalues[j].append(word_ranks[j][i][word_indices[j][i]])
    word_qvalues = [torch.stack(item) for item in word_qvalues]
    word_indices = [to_pt(item, use_cuda) for item in word_indices]
    word_indices = [item.unsqueeze(-1) for item in word_indices]  # list of batch x 1
    return word_qvalues, word_indices


def choose_command(word_ranks, word_masks_np, use_cuda, epsilon=0.0):
    batch_size = word_ranks[0].size(0)
    word_qvalues, word_indices_maxq = _choose_maxQ_command(word_ranks, word_masks_np, use_cuda)
    if epsilon > 0.0:
        _, word_indices_random = _choose_random_command(word_ranks, word_masks_np, use_cuda)
        # random number for epsilon greedy
        rand_num = np.random.uniform(low=0.0, high=1.0, size=(batch_size, 1))
        less_than_epsilon = (rand_num < epsilon).astype("float32")  # batch
        greater_than_epsilon = 1.0 - less_than_epsilon
        less_than_epsilon = to_pt(less_than_epsilon, use_cuda, type='float')
        greater_than_epsilon = to_pt(greater_than_epsilon, use_cuda, type='float')
        less_than_epsilon, greater_than_epsilon = less_than_epsilon.long(), greater_than_epsilon.long()
        chosen_indices = [
            less_than_epsilon * idx_random + greater_than_epsilon * idx_maxq for idx_random, idx_maxq in
            zip(word_indices_random, word_indices_maxq)]
    else:
        chosen_indices = word_indices_maxq
    chosen_indices = [item.detach() for item in chosen_indices]
    return word_qvalues, chosen_indices


class WordVocab:
    def __init__(self, vocab_file="./vocab.txt"):
        with open(vocab_file) as f:
            self.word_vocab = f.read().split("\n")
        self.word2id = {}
        for i, w in enumerate(self.word_vocab):
            self.word2id[w] = i
        self.EOS_id = self.word2id["</S>"]
        self.single_word_verbs = set(["inventory", "look"])
        self.preposition_map = {"take": "from",
                                "chop": "with",
                                "slice": "with",
                                "dice": "with",
                                "cook": "with",
                                "insert": "into",
                                "put": "on"}
        # FROM extras.command_templates:
        #   'inventory',
        #   'look',
        #   'prepare meal',
        #   'go east', 'go north', 'go south', 'go west',
        #   'cook {f} with {oven}',
        #   'cook {f} with {stove}',
        #   'cook {f} with {toaster}',
        #   'chop {f} with {o}',
        #   'dice {f} with {o}',
        #   'slice {f} with {o}',
        #   'lock {c|d} with {k}',
        #   'unlock {c|d} with {k}',
        #   'close {c|d}',
        #   'open {c|d}',
        #   'take {o} from {c|s}',
        #   'insert {o} into {c}',
        #   'put {o} on {s}',
        #   'drop {o}',
        #   'take {o}',
        #   'drink {f}',
        #   'eat {f}',
        #   'examine {o|t}',

        # self.word_masks_np = [verb_mask, adj_mask, noun_mask, second_adj_mask, second_noun_mask]
        self.word_masks_np = []  # will be a list of np.array (x5), one for each potential output word

    def init_with_infos(self, infos):
        # get word masks
        batch_size = len(infos["verbs"])
        mask_shape = (batch_size, len(self.word_vocab))
        verb_mask = np.zeros(mask_shape, dtype="float32")
        noun_mask = np.zeros(mask_shape, dtype="float32")
        adj_mask = np.zeros(mask_shape, dtype="float32")

        verbs_word_lists = infos["verbs"]
        # print("batch_size=", batch_size)
        # print('verbs_word_list:', verbs_word_list)
        noun_word_lists, adj_word_lists = [], []
        for entities in infos["entities"]:
            tmp_nouns, tmp_adjs = [], []
            for name in entities:
                split = name.split()
                tmp_nouns.append(split[-1])
                if len(split) > 1:
                    tmp_adjs += split[:-1]
            noun_word_lists.append(list(set(tmp_nouns)))
            adj_word_lists.append(list(set(tmp_adjs)))

        for i in range(batch_size):
            for w in verbs_word_lists[i]:
                if w in self.word2id:
                    verb_mask[i][self.word2id[w]] = 1.0
            for w in noun_word_lists[i]:
                if w in self.word2id:
                    noun_mask[i][self.word2id[w]] = 1.0
            for w in adj_word_lists[i]:
                if w in self.word2id:
                    adj_mask[i][self.word2id[w]] = 1.0
        second_noun_mask = copy.copy(noun_mask)
        second_adj_mask = copy.copy(adj_mask)
        second_noun_mask[:, self.EOS_id] = 1.0
        adj_mask[:, self.EOS_id] = 1.0
        second_adj_mask[:, self.EOS_id] = 1.0
        self.word_masks_np = [verb_mask, adj_mask, noun_mask, second_adj_mask, second_noun_mask]

    def _word_ids_to_commands(self, verb, adj, noun, adj_2, noun_2):
        """
        Turn the 5 indices into actual command strings.

        Arguments:
            verb: Index of the guessing verb in vocabulary
            adj: Index of the guessing adjective in vocabulary
            noun: Index of the guessing noun in vocabulary
            adj_2: Index of the second guessing adjective in vocabulary
            noun_2: Index of the second guessing noun in vocabulary
        """
        # turns 5 indices into actual command strings
        if self.word_vocab[verb] in self.single_word_verbs:
            return self.word_vocab[verb]
        if adj == self.EOS_id:
            res = self.word_vocab[verb] + " " + self.word_vocab[noun]
        else:
            res = self.word_vocab[verb] + " " + self.word_vocab[adj] + " " + self.word_vocab[noun]
        if self.word_vocab[verb] not in self.preposition_map:
            return res
        if noun_2 == self.EOS_id:
            return res
        prep = self.preposition_map[self.word_vocab[verb]]
        if adj_2 == self.EOS_id:
            res = res + " " + prep + " " + self.word_vocab[noun_2]
        else:
            res = res + " " + prep + " " + self.word_vocab[adj_2] + " " + self.word_vocab[noun_2]
        return res

    def get_chosen_strings(self, chosen_indices):
        """
        Turns list of word indices into actual command strings.

        Arguments:
            chosen_indices: Word indices chosen by model.
        """
        chosen_indices_np = [to_np(item)[:, 0] for item in chosen_indices]
        res_str = []
        batch_size = chosen_indices_np[0].shape[0]
        for i in range(batch_size):
            verb, adj, noun, adj_2, noun_2 = chosen_indices_np[0][i],\
                                             chosen_indices_np[1][i],\
                                             chosen_indices_np[2][i],\
                                             chosen_indices_np[3][i],\
                                             chosen_indices_np[4][i]
            res_str.append(self._word_ids_to_commands(verb, adj, noun, adj_2, noun_2))
        return res_str


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

class CustomAgent:
    def __init__(self):
        """
        Arguments:
            word_vocab: List of words supported.
        """
        self.mode = "train"
        with open("config.yaml") as reader:
            self.config = yaml.safe_load(reader)
        self.vocab = WordVocab(vocab_file="./vocab.txt")
        self.batch_size = self.config['training']['batch_size']
        self.max_nb_steps_per_episode = self.config['training']['max_nb_steps_per_episode']
        self.nb_epochs = self.config['training']['nb_epochs']
        self._debug_quit = False

        # Set the random seed manually for reproducibility.
        seedval = self.config['general']['random_seed']
        np.random.seed(seedval)
        torch.manual_seed(seedval)
        if torch.cuda.is_available():
            if not self.config['general']['use_cuda']:
                print("WARNING: CUDA device detected but 'use_cuda: false' found in config.yaml")
                self.use_cuda = False
            else:
                torch.backends.cudnn.deterministic = True
                torch.cuda.manual_seed(seedval)
                self.use_cuda = True
        else:
            self.use_cuda = False

        self.agentNN = AgentDQN(self.config, self.vocab, self.use_cuda)

        self.save_frequency = self.config['checkpoint']['save_frequency']


        # epsilon greedy
        self.epsilon_anneal_episodes = self.config['general']['epsilon_anneal_episodes']
        self.epsilon_anneal_from = self.config['general']['epsilon_anneal_from']
        self.epsilon_anneal_to = self.config['general']['epsilon_anneal_to']
        self.epsilon = self.epsilon_anneal_from
        self.update_per_k_game_steps = self.config['general']['update_per_k_game_steps']

        # self.nlp = spacy.load('en_core_web_lg', disable=['ner', 'parser', 'tagger']) #spacy used only for tokenization
        self.nlp = None
        self.current_episode = 0
        self.current_step = 0
        self._episode_has_started = False
        self.history_avg_scores = HistoryScoreCache(capacity=1000)
        self.best_avg_score_so_far = 0.0

    def train(self):
        """
        Tell the agent that it's training phase.
        """
        self.mode = "train"
        self.agentNN.model.train()


    def eval(self):
        """
        Tell the agent that it's evaluation phase.
        """
        self.mode = "eval"
        self.agentNN.model.eval()

    def _start_episode(self, obs: List[str], infos: Dict[str, List[Any]]) -> None:
        """
        Prepare the agent for the upcoming episode.

        Arguments:
            obs: Initial feedback for each game.
            infos: Additional information for each game.
        """
        self.init_with_infos(obs, infos)
        self._episode_has_started = True

    def _end_episode(self, obs: List[str], scores: List[int], infos: Dict[str, List[Any]]) -> None:
        """
        Tell the agent the episode has terminated.

        Arguments:
            obs: Previous command's feedback for each game.
            score: The score obtained so far for each game.
            infos: Additional information for each game.
        """
        self.finish()
        self._episode_has_started = False

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
        request_infos = EnvInfos()
        request_infos.description = True
        request_infos.inventory = True
        request_infos.entities = True
        request_infos.verbs = True
        request_infos.extras = ["recipe"]
        request_infos.facts = True
        request_infos.location = True
        return request_infos

    def init_with_infos(self, obs: List[str], infos: Dict[str, List[Any]]):
        """
        Prepare the agent for the upcoming games.

        Arguments:
            obs: Previous command's feedback for each game.
            infos: Additional information for each game.
        """
        # reset agent, get vocabulary masks for verbs / adjectives / nouns
        self.scores = []
        self.dones = []
        self.prev_actions = ["" for _ in range(len(obs))]
        self.agents = []
        for idx in range(len(obs)):
            if 'game_id' in infos:
                game_id = parse_gameid(infos['game_id'][idx])
            else:
                game_id = str(idx)
            self.agents.append(
                NailAgent(
                    self.config['general']['random_seed'],  #seed
                    "TW",     # rom_name
                    game_id)  # env_name
            )

        self.vocab.init_with_infos(infos)
        self.prev_obs = obs
        self.cache_description_id_list = None   # similar to .prev_obs
        self.cache_chosen_indices = None        # similar to .prev_actions
        self.current_step = 0


    def get_game_step_info(self, obs: List[str], infos: Dict[str, List[Any]]):
        """
        Get all the available information, and concat them together to be tensor for
        a neural model. we use post padding here, all information are tokenized here.

        Arguments:
            obs: Previous command's feedback for each game.
            infos: Additional information for each game.
        """
        word2id = self.vocab.word2id
        inventory_id_list = get_token_ids_for_items(infos["inventory"], word2id, tokenizer=self.nlp)

        feedback_id_list = get_token_ids_for_items(obs, word2id, tokenizer=self.nlp)

        quest_id_list = get_token_ids_for_items(infos["extra.recipe"], word2id, tokenizer=self.nlp)

        prev_action_id_list = get_token_ids_for_items(self.prev_actions, word2id, tokenizer=self.nlp)

        description_id_list = get_token_ids_for_items(infos["description"],
                                                      word2id, tokenizer=self.nlp, subst_if_empty=['end'])

        description_id_list = [_d + _i + _q + _f + _pa for (_d, _i, _q, _f, _pa) in zip(description_id_list,
                                                                                        inventory_id_list,
                                                                                        quest_id_list,
                                                                                        feedback_id_list,
                                                                                        prev_action_id_list)]

        input_description = pad_sequences(description_id_list, maxlen=max_len(description_id_list)).astype('int32')
        input_description = to_pt(input_description, self.use_cuda)

        return input_description, description_id_list

    def act_eval(self, obs: List[str], scores: List[int], dones: List[bool], infos: Dict[str, List[Any]]) -> List[str]:
        """
        Acts upon the current list of observations, during evaluation.

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
        use_nail_agent = True

        if self.current_step > 0:
            # append scores / dones from previous step into memory
            # Update the agent.
            if use_nail_agent:
                for idx, agent in enumerate(self.agents):
                    prevobs = self.prev_obs
                    prevaction = self.prev_actions[idx]
                    agent.observe(prevobs[idx], prevaction, scores[idx], obs[idx], dones[idx])
                    # Output this step.
                    print("<Step {}> [{}]{}  Action: [{}]   Score: {}\nobs::".format(
                        self.current_step,
                        idx, agent.env_name if hasattr(agent, 'env_name') else '',
                        prevaction, scores[idx],)) # obs[idx]))
                self.prev_obs = obs  # save for constructing transition during next step
            self.scores.append(scores)
            self.dones.append(dones)

        if all(dones):
            self._end_episode(obs, scores, infos)
            return  # Nothing to return.

        if use_nail_agent:
            chosen_strings = []
            agent_id = 0
            game_id = None
            for idx, (desctext, agent) in enumerate(zip(obs, self.agents)):
                assert idx == agent_id
                print("--- current step: {} -- NAIL[{}]: observation=[{}]".format(self.current_step, agent_id, desctext))
                if 'inventory' in infos:
                    print("\tINVENTORY:", infos['inventory'][agent_id])
                if 'game_id' in infos:
                    print("infos[game_id]=", infos['game_id'][agent_id])
                    #CHEAT on one specific game (to debug some new code)
                # if self.current_step == 0:
                #     actiontxt = "enable print state option"  # this HACK works even without env.activate_state_tracking()
                if 'facts' in infos:
                    world_facts = infos['facts'][agent_id]
                    verbose = (self.current_step == 0)
                    if agent.gt_nav == agent.active_module and agent.gt_nav._path_idx == len(agent.gt_nav.path):
                        verbose = True
                    observable_facts, player_room = filter_observables(world_facts, verbose=verbose)
                    print("FACTS IN SCOPE:")
                    for fact in observable_facts:
                        print('\t', fact)
                        # print_fact(game, fact)

                    agent.set_ground_truth(world_facts)
                    if self.current_step == 0:
                        # agent.gt_navigate("kitchen")  # set nav destination (using Ground Truth knowledge)
                        agent.gi.event_stream.push(NeedToGoTo('kitchen', groundtruth=True))
                        # if infos['game_id'][agent_id] == 'tw-cooking-recipe1+open+go6-qqqrhLbXf7bOTRoa.ulx':
                        #     agent.gi.event_stream.push(NeedToAcquire(objnames=['block of cheese'], groundtruth=True))
                            # agent.modules[0].add_required_obj('block of cheese')
                    #     actiontxt = "go west"
                    # elif self.current_step == 1:
                    #     actiontxt = "go north"
                    # else:
                    if self._debug_quit and self._debug_quit == self.current_step:
                        dbg("[{}] DEBUG QUIT! step={}".format(agent_id, self.current_step))
                        # exit(0)
                    # if self.current_step > 2 and player_room.name == 'kitchen' and not self._debug_quit:
                    #     self._debug_quit = self.current_step + 2  # early abort after executing one more action
                    if self.current_step > 0 and observable_facts:
                        # TODO: CLEANUP - this should really be done as part of/after agent.observe(), above
                        # world = World.from_facts(facts)
                        # add obs_facts to our KnowledgeGraph (self.gi.kg)
                        agent.gi.kg.update_facts(observable_facts, prev_action=self.prev_actions[agent_id])

                actiontxt = agent.choose_next_action(desctext)
                print("NAIL[{}] choose_next_action -> {}".format(agent_id, actiontxt))

                chosen_strings.append(actiontxt)
                agent_id += 1
        else:
            input_description, _ = self.get_game_step_info(obs, infos)
            word_ranks = self.agentNN.infer_word_ranks(input_description)  # list of batch x vocab
            _, word_indices_maxq = _choose_maxQ_command(word_ranks, self.vocab.word_masks_np, self.use_cuda)
            chosen_indices = word_indices_maxq
            chosen_indices = [item.detach() for item in chosen_indices]
            chosen_strings = self.vocab.get_chosen_strings(chosen_indices)
        self.prev_actions = chosen_strings
        self.current_step += 1

        return chosen_strings

    def act(self, obs: List[str], scores: List[int], dones: List[bool], infos: Dict[str, List[Any]]) -> List[str]:
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
        if not self._episode_has_started:
            self._start_episode(obs, infos)

        if self.mode == "eval":
            return self.act_eval(obs, scores, dones, infos)

        if self.current_step > 0:
            # append scores / dones from previous step into memory
            self.scores.append(scores)
            self.dones.append(dones)
            if self.mode == "eval":
                if all(dones):
                    self._end_episode(obs, scores, infos)
                    return  # Nothing to return.
            elif self.mode == "train":
                # compute previous step's rewards and masks
                rewards_np, mask_np = self.compute_reward()
                mask_pt = to_pt(mask_np, self.use_cuda, type='float')
                rewards_pt = to_pt(rewards_np, self.use_cuda, type='float')

        input_description, description_id_list = self.get_game_step_info(obs, infos)
        # generate commands for one game step, epsilon greedy is applied, i.e.,
        # there is epsilon of chance to generate random commands
        word_ranks = self.agentNN.infer_word_ranks(input_description)  # list of batch x vocab
        assert word_ranks[0].size(0) == input_description.size(0)   # refactoring

        _, chosen_indices = choose_command(word_ranks,
                                           self.vocab.word_masks_np,
                                           self.use_cuda,
                                           epsilon=(0.0 if self.mode == "eval" else self.epsilon))
        chosen_strings = self.vocab.get_chosen_strings(chosen_indices)
        self.prev_actions = chosen_strings

        if self.mode == "train":
            # push info from previous game step into replay memory
            if self.current_step > 0:
                for b in range(len(obs)):
                    if mask_np[b] == 0:
                        continue
                    is_prior = rewards_np[b] > 0.0
                    # Transition = namedtuple('Transition', ('observation_id_list', 'word_indices',
                    #                                        'reward', 'mask', 'done',
                    #                                        'next_observation_id_list',
                    #                                        'next_word_masks'))

                    transition = Transition(
                                            self.cache_description_id_list[b],
                                            [item[b] for item in self.cache_chosen_indices],
                                            rewards_pt[b],
                                            mask_pt[b],
                                            dones[b],
                                            description_id_list[b],
                                            [word_mask[b] for word_mask in self.vocab.word_masks_np])

                    self.agentNN.replay_memory.push(is_prior=is_prior, transition=transition)

            # cache new info in current game step into caches
            self.cache_description_id_list = description_id_list
            self.cache_chosen_indices = chosen_indices

            # update neural model by replaying snapshots in replay memory
            if self.current_step > 0 and self.current_step % self.update_per_k_game_steps == 0:
                loss = self.agentNN.update()
                if loss is not None:
                    # Backpropagate
                    self.agentNN.backpropagate(loss)

        self.current_step += 1

        if all(dones):
            self._end_episode(obs, scores, infos)
            return  # Nothing to return.
        return chosen_strings

    def compute_reward(self):
        """
        Compute rewards by agent. Note this is different from what the training/evaluation
        scripts do. Agent keeps track of scores and other game information for training purpose.

        """
        # mask = 1 if game is not finished or just finished at current step
        if len(self.dones) == 1:
            # it's not possible to finish a game at 0th step
            mask = [1.0 for _ in self.dones[-1]]
        else:
            assert len(self.dones) > 1
            mask = [1.0 if not self.dones[-2][i] else 0.0 for i in range(len(self.dones[-1]))]
        mask = np.array(mask, dtype='float32')
        # rewards returned by game engine are always accumulated value the
        # agent have recieved. so the reward it gets in the current game step
        # is the new value minus values at previous step.
        rewards = np.array(self.scores[-1], dtype='float32')  # batch
        if len(self.scores) > 1:
            prev_rewards = np.array(self.scores[-2], dtype='float32')
            rewards = rewards - prev_rewards
        return rewards, mask

    def finish(self) -> None:
        """
        All games in the batch are finished. One can choose to save checkpoints,
        evaluate on validation set, or do parameter annealing here.

        """
        # Game has finished (either win, lose, or exhausted all the given steps).
        self.final_rewards = np.array(self.scores[-1], dtype='float32')  # batch
        dones = []
        for d in self.dones:
            d = np.array([float(dd) for dd in d], dtype='float32')
            dones.append(d)
        dones = np.array(dones)
        step_used = 1.0 - dones
        self.step_used_before_done = np.sum(step_used, 0)  # batch

        self.history_avg_scores.push(np.mean(self.final_rewards))
        # save checkpoint
        if self.mode == "train" and self.current_episode % self.save_frequency == 0:
            avg_score = self.history_avg_scores.get_avg()
            if avg_score > self.best_avg_score_so_far:
                self.best_avg_score_so_far = avg_score

                self.agentNN.save_checkpoint(self.current_episode)

        self.current_episode += 1
        # annealing
        if self.current_episode < self.epsilon_anneal_episodes:
            self.epsilon -= (self.epsilon_anneal_from - self.epsilon_anneal_to) / float(self.epsilon_anneal_episodes)
