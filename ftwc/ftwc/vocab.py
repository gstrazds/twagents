import copy
from typing import List
import numpy as np
import spacy
import gym.spaces

from .generic import to_np

_global_nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'tagger'])  # used only for tokenization




def _ensure_padded_len(list_of_ids_in: List[int], pad_to_len: int, pad_value: int, eos_value: int = None) -> List[int]:
    delta_len = pad_to_len - len(list_of_ids_in)
    list_of_ids = list_of_ids_in.copy()
    if delta_len > 0:
        list_of_ids.extend([pad_value] * delta_len)   # pad to desired length

    if eos_value is not None and eos_value not in list_of_ids and pad_value in list_of_ids:
        if len(list_of_ids) and list_of_ids[-1] == pad_value:
            end_idx = len(list_of_ids)-1
            while end_idx > 0 and list_of_ids[end_idx-1] == pad_value:
                end_idx -= 1
            list_of_ids[end_idx] = eos_value

    if delta_len < 0:
        return list_of_ids[:pad_to_len]
    return list_of_ids


class WordVocab:

    MAX_NUM_OBS_TOKENS = 768   # note: max words in a room description for ftwc training games is approx 243
    # (but not counting spaces or punct)
    # also inventory and other feedback can dynamically be added to the raw description

    MAX_NUM_CMD_TOKENS = 8

    def __init__(self, vocab_file="./vocab.txt"):
        with open(vocab_file) as f:
            self.word_vocab = f.read().split("\n")
        self.word2id = {}
        # self.id2word = []  # same as self.word_vocab
        for i, w in enumerate(self.word_vocab):
            self.word2id[w] = i
            # self.id2word[i] = w  # same as self.word_vocab
        self.PAD = "<PAD>"    # excluded when random sampling
        self.UNK = "<UNK>"
        self.BOS = "<S>"
        self.EOS = "</S>"
        self.SEP = "<|>"

        self.EOS_id = self.word2id[self.EOS]
        self.UNK_id = self.word2id.get(self.UNK, self.word2id.get("<unk>", 1))  # default: 1 if neither "<UNK>" or "<unk>"
        self.PAD_id = self.word2id.get(self.PAD, self.word2id.get("<pad>", 0))  # should be 0
        assert self.PAD_id == 0
        self.single_word_verbs = set(["inventory", "look", "north", "south", "east", "west", "wait"])
        self.single_word_nouns = ["knife", "oven", "stove", "bbq", "grill"]
        self.preposition_map = {
                                #"take": ("from", ),  #for FTWC, can simply "take" without specifying from where
                                "chop": ("with", ),
                                "slice": ("with", ),
                                "dice": ("with", ),
                                "cook": ("with", ),
                                "insert": ("into", ),
                                "put": ("on", "in", "into"),
                                "unlock": ("with", ),
                                }
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

    def preproc_tw_string(self, s, str_type='None', tokenizer=None, lower_case=True):
        if s is None:
            return ["nothing"]
        s = s.replace("\n", ' ')
        if s.strip() == "":
            return ["nothing"]
        if str_type == 'feedback':
            if "$$$$$$$" in s:
                s = ""
            if "-=" in s:
                s = s.split("-=")[0]
        s = s.strip()
        if len(s) == 0:
            return ["nothing"]
        tokens = [t.text for t in tokenizer(s)]
        if lower_case:
            tokens = [t.lower() for t in tokens]
        return tokens

    def _word_ids_to_command_str(self, verb, adj, noun, adj_2, noun_2, strip_padding=True) -> str:
        """
        Turn the 5 model-generated indices into an actual command string.

        Arguments:
            verb: Index of the guessing verb in vocabulary
            adj: Index of the guessing adjective in vocabulary
            noun: Index of the guessing noun in vocabulary
            adj_2: Index of the second guessing adjective in vocabulary
            noun_2: Index of the second guessing noun in vocabulary
        """
        # turns 5 indices into an actual command string

        if self.word_vocab[verb] in self.single_word_verbs:
            return self.word_vocab[verb]
        if adj == self.EOS_id or strip_padding and adj == self.PAD_id:
            res = self.word_vocab[verb] + " " + self.word_vocab[noun]
        elif noun == self.EOS_id and adj != self.PAD_id:
            res = self.word_vocab[verb] + " " + self.word_vocab[adj]
        else:
            res = self.word_vocab[verb] + " " + self.word_vocab[adj] + " " + self.word_vocab[noun]
        if self._tokid_to_word(verb) in self.preposition_map:
            adj_x = None
            if self._tokid_to_word(noun_2) in self.single_word_nouns and adj_2 != self.PAD_id:
                adj_x = noun
                noun = adj_2
                adj_2 = self.PAD_id
                res = f"{self.word_vocab[verb]} {self.word_vocab[adj]} {self.word_vocab[adj_x]} {self.word_vocab[noun]}"
            # use first available preposition. TODO: adapt based on object type of arg2
            prep = self.preposition_map[self.word_vocab[verb]][0]
            if adj_2 == self.EOS_id or strip_padding and adj_2 == self.PAD_id:
                res = res + " " + prep + " " + self.word_vocab[noun_2]
            else:
                res = res + " " + prep + " " + self.word_vocab[adj_2] + " " + self.word_vocab[noun_2]
        else:
            if adj_2 == self.EOS_id:
                return res
            elif not(strip_padding and adj_2 == self.PAD_id):
                res = res + " " + self.word_vocab[adj_2]
            if noun_2 == self.EOS_id:
                return res
            elif not(strip_padding and noun_2 == self.PAD_id):
                res = res + " " + self.word_vocab[noun_2]
        return res

    def get_chosen_strings(self, chosen_indices, strip_padding=False):
        """
        Turns list of word indices into actual command strings.

        Arguments:
            chosen_indices: Word indices chosen by model.
        """
        chosen_indices_np = [to_np(item)[:, 0] for item in chosen_indices]
        ret_strings = []
        batch_size = chosen_indices_np[0].shape[0]
        for i in range(batch_size):
            verb, adj, noun, adj_2, noun_2 = chosen_indices_np[0][i],\
                                             chosen_indices_np[1][i],\
                                             chosen_indices_np[2][i],\
                                             chosen_indices_np[3][i],\
                                             chosen_indices_np[4][i]
            ret_strings.append(
                self._word_ids_to_command_str(verb, adj, noun, adj_2, noun_2, strip_padding=strip_padding))
        return ret_strings

    def _tokid_to_word(self, tokid):
        if tokid >=0 and tokid < len(self.word_vocab):
            return self.word_vocab[tokid]
        return self.UNK

    def _words_to_ids(self, words: List[str]) -> List[int]:
        return [self.word2id.get(word, self.UNK_id) for word in words]
        # ids = []
        # for word in words:
            # try:
            #   ids.append(self.word2id.get(word, self.UNK_id))
            # except KeyError:
            #     ids.append(1)
        # return ids

    def token_id_lists_from_strings(self, str_list: List[str], str_type=None, subst_if_empty=None):
        token_lists = [self.preproc_tw_string(item, str_type=str_type, tokenizer=self.get_tokenizer())
                       for item in str_list]
        for i, tokens in enumerate(token_lists):
            if subst_if_empty and not tokens:  #len(d) == 0:
                token_lists[i] = subst_if_empty  # if empty description, insert replacement (list of tokens)
            elif str_type == 'cmd':
                while 'the' in tokens:
                    tokens.pop(tokens.index('the'))
                # print("*********tokens=", tokens)
                if tokens[0] in self.preposition_map.keys():
                    for preposition in self.preposition_map[tokens[0]]:
                        if preposition in tokens:   # adjust the token list to parse (adj noun) phrases
                            # print("!!! FOUND PREP", preposition)
                            prep_idx = tokens.index(preposition)
                            tokens.pop(prep_idx)   # remove the preposition from the token list
                            if prep_idx == 2:
                                tokens.insert(1, self.PAD)  # arg1 must be noun with no adj
                            cmd_len = len(tokens)
                            if cmd_len == 5 and tokens[-1] == self.EOS or cmd_len == 4:
                                tokens.insert(3, self.PAD)  # arg2 must be noun with no adj
                elif tokens[0] not in self.single_word_verbs:  # then should be a verb with one arg (noun phrase)
                    if len(tokens) == 3 and tokens[2] == self.EOS or len(tokens) == 2:
                        tokens.insert(1, self.PAD)  # arg1 - a noun with no adj

        id_lists = [self._words_to_ids(tokens) for tokens in token_lists]
        return id_lists

    def command_strings_to_np(self, command_strings, pad_to_len=5, pad=0):   # pad token id
        # TODO: REWRITE THIS!
        list_of_id_lists = self.token_id_lists_from_strings(command_strings, str_type='cmd')
        # assert len(oracle_word_indices) == batch_size  # list (len=n_batch) of lists (token ids)
        list_of_padded_idlists = [_ensure_padded_len(idlist, pad_to_len, pad) for idlist in list_of_id_lists]
        return_list = [ [] for _ in range(pad_to_len)]
        for idlist in list_of_padded_idlists:
            for i in range(pad_to_len):
                return_list[i].append(idlist[i])
        for i in range(len(return_list)):
            return_list[i] = np.array(return_list[i])
        return return_list

    def get_tokenizer(self):
        # if not self.nlp:    #TODO: this should be a per-process singleton
        #     import spacy
        #     self.nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'tagger']) #spacy used only for tokenization
        # return self.nlp
        return _global_nlp


class VocabularyHasDuplicateTokens(ValueError):
    pass


class WordSpace(gym.spaces.MultiDiscrete):  # adapted from textworld.gym.text_spaces
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

    def __init__(self, max_length, vocab):
        """
        Parameters
        ----------
        max_length : int
            Maximum number of words in a text.
        vocab : list of strings
            Vocabulary defining this space. It shouldn't contain any
            duplicate words.
        """
        # if len(vocab) != len(set(vocab)):
        #     raise VocabularyHasDuplicateTokens()

        self.PAD = "<PAD>"    # excluded when random sampling
        self.UNK = "<UNK>"
        self.BOS = "<S>"
        self.EOS = "</S>"
        self.SEP = "<|>"
        self.max_length = max_length
        special_tokens = [self.PAD, self.UNK, self.EOS, self.BOS, self.SEP]
        self.sampling_offset = len(special_tokens)
        self.vocab_size = len(vocab) - self.sampling_offset
        # self.vocab = [w for w in special_tokens if w not in vocab]
        # self.vocab += list(vocab)
        # self.vocab_set = set(self.vocab)  # For faster lookup.
        # self.vocab_size = len(self.vocab)
        # self.id2w = {i: w for i, w in enumerate(self.vocab)}
        # self.w2id = {w: i for i, w in self.id2w.items()}
        # self.PAD_id = self.w2id[self.PAD]
        # self.UNK_id = self.w2id[self.UNK]
        # self.BOS_id = self.w2id[self.BOS]
        # self.EOS_id = self.w2id[self.EOS]
        # self.SEP_id = self.w2id[self.SEP]
        # super().__init__([len(self.vocab) - 1] * self.max_length)
        super().__init__([self.vocab_size] * self.max_length)
        self.dtype = np.int64  # Overwrite Gym's dtype=int8.
