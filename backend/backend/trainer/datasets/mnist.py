import pytorch_lightning as pl
import torch
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from functools import partial


class MNISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./",
        batch_size: int = 32,
        num_workers: int = 2,
        seed: int = 42,
        train_ratio: float = 0.9
    ):
        super().__init__()
        self.in_channels = 1
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_ratio = train_ratio
        self.val_ratio = 1 - train_ratio
        self.seed = seed
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        self.loader = partial(
            DataLoader,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=True
        )

    def setup(self, stage: str):
        mnist_partial = partial(
            MNIST,
            root=self.data_dir, transform=self.transform
        )
        retrying = True
        while retrying:
            try:
                mnist_partial(train=True, download=True)
                mnist_partial(train=False, download=True)
                retrying = False
            except:
                pass

        if stage == "fit":
            mnist_full = mnist_partial(train=True)
            self.mnist_train, self.mnist_val = random_split(
                dataset=mnist_full,
                lengths=[self.train_ratio, self.val_ratio],
                generator=torch.Generator().manual_seed(self.seed)
            )
        else:
            self.mnist_test = mnist_partial(train=False)

    def train_dataloader(self):
        return self.loader(dataset=self.mnist_train, shuffle=True)

    def val_dataloader(self):
        return self.loader(dataset=self.mnist_val)

    def test_dataloader(self):
        return self.loader(dataset=self.mnist_test)
