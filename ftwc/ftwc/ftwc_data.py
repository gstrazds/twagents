from typing import List, Optional, Tuple, Any
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

from .buffers import Transition


def training_batch_generator(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    logic for generating a new batch of data (from the replay buffer to be passed to the DataLoader

    Returns:
        yields a Experience tuple containing the state, action, reward, done and next_state.
    """
    episode_reward = 0
    episode_steps = 0

    while True:
        self.total_steps += 1
        episode_steps += 1
        # choose next action: AgentDQN.select_next_action()
        # action = self.agent(self.state, self.device)
        # step environment and observe results: AgentDQN.training_step()
        # next_state, r, is_done, _ = self.env.step(action[0])
        # episode_reward += r
        # add transition to replay buffer: AgentDQN.save_transition_for_replay()
        # exp = Experience(state=self.state, action=action[0], reward=r, done=is_done, new_state=next_state)
        # exp = Transition(observation_id_list=,
        #                  word_indices=,
        #                  reward=r,
        #                  mask=,
        #                  done=is_done,
        #                  next_observation_id_list=,
        #                  next_word_masks=)
        #
        # self.agent.update_epsilon(self.global_step)
        # self.buffer.append(exp)
        # self.state = next_state

        if is_done:
            self.done_episodes += 1
            self.total_rewards.append(episode_reward)
            self.total_episode_steps.append(episode_steps)
            self.avg_rewards = float(
                np.mean(self.total_rewards[-self.avg_reward_len:])
            )
            self.state = self.env.reset()
            episode_steps = 0
            episode_reward = 0

        states, actions, rewards, dones, new_states = self.buffer.sample(self.batch_size)

        for idx, _ in enumerate(dones):
            yield states[idx], actions[idx], rewards[idx], dones[idx], new_states[idx]

        # Simulates epochs
        if self.total_steps % self.batches_per_epoch == 0:
            break


class GameStepDataset(torch.utils.data.Dataset):
    """PyTorch dataset of First TextWorld Challenge (cooking) games"""

    def __init__(self, num_steps=2000):
        self.num_steps = num_steps

    def __len__(self):
        return self.num_steps

    def __getitem__(self, idx):
        """Generates one sample of data"""
        # Select sample
        return idx

class GamefileDataset(torch.utils.data.Dataset):
    """PyTorch dataset of First TextWorld Challenge (cooking) games"""

    def __init__(self, games_list: List[str]):
        self.gamefiles = games_list

    def __len__(self):
        return len(self.gamefiles)

    def __getitem__(self, idx):
        """Generates one sample of data"""
        # Select sample
        gamefile = self.gamefiles[idx]
        return gamefile

        # # Load data and get label
        # X = torch.load('data/' + ID + '.pt')
        # y = self.labels[ID]
        # return X, y


class GamefileDataModule(pl.LightningDataModule):
    def __init__(self, gamefiles: List[str], testfiles : Optional[List[str]] = None, **kwargs):
        super().__init__()
        self._training_list = gamefiles
        self._testing_list = testfiles
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def setup(self, stage: Optional[str] = None):
        # NOTE/WARNING: setup is called from every GPU. Setting state here is okay.

        # Assign Train/val split(s) for use in Dataloaders
        if stage == 'fit' or stage is None:
            gamefiles_ds = GamefileDataset(self._training_list)
            total = len(gamefiles_ds)
            if total > 1:
                val_len = max(1, int(0.1 * total))
                train_len = total - val_len
                _, self.val_ds = random_split(gamefiles_ds, [train_len, val_len])
                self.train_ds = GameStepDataset()
            else:  # if we only have one training game, use it for both training and validation
                self.train_ds = GameStepDataset()
                self.val_ds = gamefiles_ds
            # self.dims = self.mnist_train[0][0].shape

        # Assign Test split(s) for use in Dataloaders
        if stage == 'test' or stage is None:
            if self._testing_list:
                self.test_ds = GamefileDataset(self._testing_list)
                # self.dims = getattr(self, 'dims', self.mnist_test[0][0].shape)

    train_params = { #'shuffle': True
       'batch_size': 1,
        'num_workers': 1,
    }
    val_params = { #'shuffle': False,
        'batch_size': 1,
        'num_workers': 1,
    }
    test_params = { #'shuffle': False,
        'batch_size': 1,
        'num_workers': 1,
    }

    def train_dataloader(self):
        return DataLoader(self.train_ds, **self.train_params)    # Parameters

    def val_dataloader(self):
        return DataLoader(self.val_ds, **self.val_params)    # Parameters

    def test_dataloader(self):
        if self.test_ds:
            return DataLoader(self.test_ds, **self.test_params)    # Parameters
        return None
