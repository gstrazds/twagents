import copy
import numpy as np
import spacy
from .generic import preproc, to_np


_global_nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'tagger'])  # used only for tokenization


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

    def init_from_infos_lists(self, verbs_word_lists, entities_word_lists):
        # get word masks
        # initialized at the start of each episode: verb_mask, noun_mask, adj_mask - specific to each game in the batch
        batch_size = len(verbs_word_lists)
        assert len(entities_word_lists) == batch_size, f"{batch_size} {len(entities_word_lists)}"
        mask_shape = (batch_size, len(self.word_vocab))
        verb_mask = np.zeros(mask_shape, dtype="float32")
        noun_mask = np.zeros(mask_shape, dtype="float32")
        adj_mask = np.zeros(mask_shape, dtype="float32")

        # print("batch_size=", batch_size)
        # print('verbs_word_list:', verbs_word_list)
        noun_word_lists, adj_word_lists = [], []
        for entities in entities_word_lists:
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

    def _words_to_ids(self, words):
        ids = []
        for word in words:
            try:
                ids.append(self.word2id[word])
            except KeyError:
                ids.append(1)
        return ids

    def get_token_ids_for_items(self, item_list, subst_if_empty=None):
        token_list = [preproc(item, tokenizer=self.get_tokenizer()) for item in item_list]
        if subst_if_empty:
            for i, d in enumerate(token_list):
                if len(d) == 0:
                    token_list[i] = subst_if_empty  # if empty description, insert replacement (list of tokens)
        id_list = [self._words_to_ids(tokens) for tokens in token_list]
        return id_list

    def get_tokenizer(self):
        # if not self.nlp:    #TODO: this should be a per-process singleton
        #     import spacy
        #     self.nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'tagger']) #spacy used only for tokenization
        # return self.nlp
        return _global_nlp

