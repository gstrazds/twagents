from typing import List, Optional, Any

import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl


class GamefileDataset(torch.utils.data.Dataset):
  'PyTorch dataset of First TextWorld Challenge (cooking) games'
  def __init__(self, games_list: List[str]):
        'Initialization'
        self.gamefiles = games_list

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.gamefiles)

  def __getitem__(self, idx):
        'Generates one sample of data'
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
                self.train_ds, self.val_ds = random_split(gamefiles_ds, [train_len, val_len])
            else:  # if we only have one training game, use it for both training and validation
                self.train_ds = gamefiles_ds
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
