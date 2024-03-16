import torch
import trainer
import torch.nn as nn
import pytorch_lightning as pl
from trainer.configs import ANNConfig


class ANNModel(pl.LightningModule):
    def __init__(
        self,
        config: ANNConfig
    ):
        super().__init__()
        self.config = config
        hidden_depth = [28*28, 64, 64, 10]
        model = nn.ModuleList([])
        for i in range(len(hidden_depth)-1):
            model.append(nn.Linear(hidden_depth[i], hidden_depth[i+1]))
            if i != len(hidden_depth)-2:
                model.append(nn.Dropout(config.dropout))
                model.append(nn.ReLU())
        self.model = nn.Sequential(*model)
        self.criterion = nn.CrossEntropyLoss()
        self.statistic = {
            "train": trainer.RunningStatistic("train"),
            "val": trainer.RunningStatistic("val"),
            "test": trainer.RunningStatistic("test")
        }

    def forward(self, x):
        return self.model(x)

    def _get_accuracy(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor
    ):
        acc = (y_true == y_pred.argmax(-1)).float().mean()
        count = y_true.shape[0]
        return acc, count

    def _get_loss(self, batch, stage: str):
        x, y = batch
        y_pred = self(x.view(x.shape[0], -1))
        # print(y_pred.shape, y.shape, y.max(), y.min())
        loss = self.criterion(y_pred, y.long())
        with torch.no_grad():
            acc, count = self._get_accuracy(y.cpu(), y_pred.cpu())
            self.statistic[stage].update_step(loss.cpu(), acc, count)

        return loss

    def training_step(self, batch, idx):
        return self._get_loss(batch, stage="train")

    def validation_step(self, batch, idx):
        return self._get_loss(batch, stage="val")

    def test_step(self, batch, idx):
        return self._get_loss(batch, stage="test")

    def on_train_epoch_end(self) -> None:
        for k in self.statistic.keys():
            self.statistic[k].update_epoch()
        print(self.statistic["train"].get_state_dict() |
              self.statistic["val"].get_state_dict() |
              self.statistic["test"].get_state_dict())

    def on_test_end(self) -> None:
        print(self.statistic["train"].get_state_dict() |
              self.statistic["val"].get_state_dict() |
              self.statistic["test"].get_state_dict())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=self.config.lr,
        )
        return [optimizer]
