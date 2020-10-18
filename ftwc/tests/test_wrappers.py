import unittest

import gym
import torch

# from pl_bolts.models.rl.common.gym_wrappers import ToTensor


class TestToTensor(unittest.TestCase):

    def setUp(self) -> None:
        # self.env = ToTensor(gym.make("CartPole-v0"))
        pass

    def test_wrapper(self):
        state = self.env.reset()
        self.assertIsInstance(state, torch.Tensor)

        new_state, _, _, _ = self.env.step(1)
        self.assertIsInstance(new_state, torch.Tensor)


if __name__ == '__main__':
    unittest.main()
