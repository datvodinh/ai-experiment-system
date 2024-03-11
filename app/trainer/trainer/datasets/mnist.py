import torch
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from functools import partial


class MNISTDataModule:
    def __init__(
        self,
        data_dir: str = "./",
        batch_size: int = 32,
        num_workers: int = 1,
        seed: int = 42,
        train_ratio: float = 0.9,
        val_ratio: float = 0.1
    ):
        self.in_channels = 1
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_ratio = min(train_ratio, 0.95)
        self.val_ratio = val_ratio
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

    def prepare_data(self) -> None:
        """Download MNIST Train and Test Dataset"""
        retry = True
        while retry:
            try:
                MNIST(self.data_dir, train=True, download=True)
                MNIST(self.data_dir, train=False, download=True)
                retry = False
            except:
                pass

    def setup(self):
        """Setup Train, Eval and Test Dataset"""
        mnist_partial = partial(
            MNIST,
            root=self.data_dir, transform=self.transform
        )
        mnist_full = mnist_partial(train=True)
        self.mnist_train, self.mnist_val, _ = random_split(
            dataset=mnist_full,
            lengths=[self.train_ratio, self.val_ratio, max(1 - self.train_ratio - self.val_ratio, 0)],
            generator=torch.Generator().manual_seed(self.seed)
        )
        self.mnist_test = mnist_partial(train=False)

    def train_dataloader(self):
        return self.loader(dataset=self.mnist_train)

    def val_dataloader(self):
        return self.loader(dataset=self.mnist_val)

    def test_dataloader(self):
        return self.loader(dataset=self.mnist_test)
