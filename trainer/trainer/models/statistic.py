import torch


class RunningStatistic:
    def __init__(self, name: str):
        self.name = name
        self.loss_step = 0
        self.acc_step = 0
        self.count_step = 0
        self.loss_epoch = [0]
        self.acc_epoch = [0]

    def get_state_dict(self):
        return {
            f"{self.name}_loss": self.loss_epoch[-1],
            f"{self.name}_acc": self.acc_epoch[-1]
        }

    def update_step(
        self,
        loss: torch.Tensor,
        acc: torch.Tensor,
        count: int
    ):
        self.acc_step = (self.acc_step * self.count_step + acc * count) / (self.count_step + count)
        self.loss_step = (self.loss_step * self.count_step + loss * count) / (self.count_step + count)
        self.count_step += count

    def reset(self):
        self.loss_step = 0
        self.acc_step = 0
        self.count_step = 0

    def update_epoch(self):
        self.loss_epoch.append(self.loss_step)
        self.acc_epoch.append(self.acc_step)
        self.reset()
