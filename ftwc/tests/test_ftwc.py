import unittest

from pytorch_lightning import Trainer, seed_everything
from ftwc.ftwc_agent import FtwcAgentLit
from ftwc.ftwc_data import GamefileDataModule

# an integration test, not part of unittest TestCase. Run separately
def test_lit_classifier():
    seed_everything(1234)

    model = FtwcAgentLit()
    train, val, test = GamefileDataModule()
    trainer = Trainer(limit_train_batches=50, limit_val_batches=20, max_epochs=2)
    trainer.fit(model, train, val)

    results = trainer.test(test_dataloaders=test)
    assert results[0]['test_acc'] > 0.7


class FtwcTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
