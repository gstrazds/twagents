import unittest
import os
import gym
import torch
import numpy as np

from ftwc.wrappers.gym_wrappers import ToTensor
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
        "open frosted-glass door":      ['open', 'frosted-glass', 'door'],
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

def generate_fake_gameid(idx=0):
    return f"Game_{idx:04d}"

def generate_tw_obs(step=0, gameid=None, idx=0):
    if gameid is None:
        gameid = generate_fake_gameid(idx)
    return f"[{gameid} (idx:{idx})] Obs {step:02d}"

class FakeTwScoreEnv(gym.Env):
    def __init__(self, batch_size=1):
        self.current_step = -1
        self.batch_size = batch_size
        self.static_infos = {'extra.uuid': [generate_fake_gameid(idx=i) for i in range(self.batch_size)]}
        self.static_infos['game_id'] = [generate_fake_gameid(idx=i) for i in range(self.batch_size)]

        # self.observation_space = spaces.Dict({
        #     name: gym.spaces.Box(shape=(2, ), low=-1, high=1, dtype=np.float32)
        #     for name in observation_keys
        # })
        # self.action_space = gym.spaces.Box(
        #     shape=(1, ), low=-1, high=1, dtype=np.float32)

    def _generate_obs_for_current_step(self):
        return [generate_tw_obs(step=self.current_step, idx=i) for i in range(self.batch_size)]

    def _generate_score_for_current_step(self):
        scores = []
        for b in range(self.batch_size):
            score = ((self.current_step + (b % 2)) % 2) * self.current_step
            if b > 1:
                score = b*self.current_step + ((b+1)%2)*b*score
            scores.append(score)
        return scores

    def reset(self):
        self.current_step = 0
        observation = self._generate_obs_for_current_step()
        return observation, self.static_infos

    def step(self, action):
        self.current_step += 1
        observation = self._generate_obs_for_current_step()
        score, terminal, info = self._generate_score_for_current_step(), [False]*self.batch_size, self.static_infos
        self.scores = score
        return observation, score, terminal, info


from ftwc.wrappers import ScoreToRewardWrapper

class TestEnvWrappers(unittest.TestCase):
    def setup(self) -> None:
        # self.env = FakeTwScoreEnv()
        pass

    def test_score2reward(self):
        batch_size = 4
        n_steps = 10
        expected_scores = [
            [1, 0, 4, 3],
            [0, 2, 4, 6],
            [3, 0, 12, 9],
            [0, 4, 8, 12],
            [5, 0, 20, 15],
            [0, 6, 12, 18],
            [7, 0, 28, 21],
            [0, 8, 16, 24],
            [9, 0, 36, 27],
            [0, 10, 20, 30]
        ]
        # _expected_rewards = [
        #     [1, -1,  3, -3,  5, -5,  7,  -7,  9,  -9],
        #     [0,  2, -2,  4, -4,  6, -6,   8, -8,  10],
        #     [4,  0,  8, -4, 12, -8, 16, -12, 20, -16],
        #     [3,  3,  3,  3,  3,  3,  3,   3,  3,   3],
        # ]
        expected_rewards = [[], [], [], []] * n_steps
        _prev_scores = [0] * batch_size
        for i in range(n_steps):
            expected_rewards[i] = [expected_scores[i][b] - _prev_scores[b] for b in range(batch_size)]
            _prev_scores = expected_scores[i]

        base_env = FakeTwScoreEnv(batch_size=batch_size)
        env = ScoreToRewardWrapper(base_env)
        reward, infos = env.reset()
        cmds = ["do something"] * batch_size
        for istep in range(n_steps):
            obs, reward, done, info = env.step(cmds)
            # print(reward, done)
            self.assertEqual(expected_scores[istep], base_env.scores)  # f"test infrastructure is broken (step:{istep})"
            for b in range(batch_size):
                # self.assertEqual(expected_rewards[istep][b], _expected_rewards[b][istep], f"step:{istep} batch_idx:{b}")
                self.assertEqual(reward[b], expected_rewards[istep][b], f"step:{istep} batch_idx:{b}")


if __name__ == '__main__':
    unittest.main()
