import unittest
import os
import gym
import torch
import numpy as np

from ftwc.gym_wrapper import ToTensor
from ftwc.vocab import WordVocab, _ensure_padded_len


VOCAB_FILE = os.path.join(os.path.dirname(__file__), '../conf/vocab.txt')


class TestVocab(unittest.TestCase):
    cmds = {
        "put purple pepper on wooden table":   ['put', 'purple', 'pepper', 'wooden', 'table'],
        "put knife on steel table":     ['put', '<PAD>', 'knife', 'steel', 'table'],
        "put chicken leg on counter":   ['put', 'chicken', 'leg', '<PAD>', 'counter'],
        "slice green cilantro with knife":   ['slice', 'green', 'cilantro', '<PAD>', 'knife'],
        "slice cilantro with knife":   ['slice', '<PAD>', 'cilantro', '<PAD>', 'knife'],
        "insert knife into fridge":     ['insert', '<PAD>', 'knife', '<PAD>', 'fridge'],
        "open toolbox":                 ['open', '<PAD>', 'toolbox', '</S>'],
        "open green box":               ['open', 'green', 'box', '</S>'],
        "go east":                      ['go', '<PAD>', 'east'],
        "north":                        ['north'],
    }

    special_cases = {
        "": ['nothing'],
        "put knife in fridge": ['put', '<PAD>', 'knife', '<PAD>', 'fridge'],
    }

    def setUp(self) -> None:
        self.cmds = TestVocab.cmds
        for k, v in self.cmds.items():   # PAD all the token lists to len = 5
            if len(v) < 5:
                if '</S>' not in v:
                    v.append('</S>')
                pad_count = 5 - len(v)
                v.extend(['<PAD>'] * pad_count)
                self.cmds[k] = v
        self.vocab = WordVocab(vocab_file=VOCAB_FILE)

    def test_decode_commands(self):
        for c, toks in self.cmds.items():
            tok_ids = self.vocab._words_to_ids(toks)
            # print(tok_ids)
            encoded_toks = [torch.tensor(np.array([[tok]])) for tok in tok_ids]
            # print(encoded_toks)
            decoded_str = self.vocab.get_chosen_strings(encoded_toks, strip_padding=True)
            self.assertEqual(len(decoded_str), 1)  # a list of strings (in this case, just one)
            decoded_str = decoded_str[0]
            self.assertEqual(c, decoded_str)

    def test_encode_commands(self):
        for c, toks in self.cmds.items():
            tok_ids = self.vocab.token_id_lists_from_strings([c], str_type='cmd')
            self.assertEqual(len(tok_ids), 1)
            tok_ids = tok_ids[0]  #list of lists, know there is only one
            tok_ids = _ensure_padded_len(tok_ids, 5, self.vocab.PAD_id, eos_value=self.vocab.EOS_id)
            # print(tok_ids)
            tok_ids = list(map(lambda t: self.vocab.word_vocab[t], tok_ids))
            # print(tok_ids)
            self.assertEqual(toks, tok_ids)



class TestToTensor(unittest.TestCase):

    def setUp(self) -> None:
        self.env = ToTensor(gym.make("CartPole-v0"))
        pass

    def test_wrapper(self):
        state = self.env.reset()
        self.assertIsInstance(state, torch.Tensor)

        new_state, _, _, _ = self.env.step(1)
        self.assertIsInstance(new_state, torch.Tensor)


if __name__ == '__main__':
    unittest.main()
