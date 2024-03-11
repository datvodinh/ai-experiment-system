import backend as be
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from backend.configs import ANNConfig


class Trainer:
    def __init__(self, config: ANNConfig):
        self.config = config
        self.model = be.ANNModel(config)
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(), lr=config.lr
        )
        self.criterion = nn.CrossEntropyLoss()
        self.datamodule = be.MNISTDataModule(batch_size=config.batch_size)
        self.datamodule.prepare_data()
        self.datamodule.setup()

        self.train_stat = be.RunningStatistic("train")
        self.val_stat = be.RunningStatistic("val")
        self.test_stat = be.RunningStatistic("test")

    def _get_accuracy(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor
    ):
        acc = (y_true == y_pred.argmax(-1)).float().mean()
        count = y_true.shape[0]
        return acc, count

    def _train_one_epoch(self):
        """Run train for one epoch"""
        self.model.train()
        for _, batch in enumerate(tqdm(self.datamodule.train_dataloader(), desc="Train")):
            x, y = batch
            x = x.flatten(1)
            y_pred = self.model(x)
            loss = self.criterion(y_pred, y)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                acc, count = self._get_accuracy(y, y_pred)
                self.train_stat.update_step(loss, acc, count)
        self.train_stat.update_epoch()

    @torch.no_grad()
    def _val_one_epoch(self):
        """Run eval for one epoch"""
        self.model.eval()
        for _, batch in enumerate(tqdm(self.datamodule.val_dataloader(), desc="Validate")):
            x, y = batch
            x = x.flatten(1)
            y_pred = self.model(x)
            loss = self.criterion(y_pred, y)
            acc, count = self._get_accuracy(y, y_pred)
            self.val_stat.update_step(loss, acc, count)
        self.val_stat.update_epoch()

    @torch.no_grad()
    def _test_one_epoch(self):
        """Run test for one epoch"""
        self.model.eval()
        for _, batch in enumerate(tqdm(self.datamodule.test_dataloader(), desc="Test")):
            x, y = batch
            x = x.flatten(1)
            y_pred = self.model(x)
            loss = self.criterion(y_pred, y)
            acc, count = self._get_accuracy(y, y_pred)
            self.test_stat.update_step(loss, acc, count)
        self.test_stat.update_epoch()

    def fit(self):
        """Fit model"""
        for _ in tqdm(range(self.config.max_epochs), desc="Fit"):
            self._train_one_epoch()
            self._val_one_epoch()
            self._test_one_epoch()


if __name__ == "__main__":
    trainer = be.Trainer(be.ANNConfig())
    trainer.fit()
