import os
import trainer as tr
import pytorch_lightning as pl

disable = False


class TrainerNode:
    def __init__(self, config: tr.ANNConfig):
        self._config = config
        self._model = tr.ANNModel(config)
        self._datamodule = tr.MNISTDataModule(batch_size=config.batch_size)
        self._trainer = pl.Trainer(
            default_root_dir=os.path.join(os.getcwd(), "checkpoints"),
            gradient_clip_val=1.0,
            max_epochs=config.max_epochs,
            deterministic=False,
            precision="16-mixed",
        )

    def fit(self):
        self._trainer.fit(model=self._model, datamodule=self._datamodule)


class Trainer:
    def __init__(self) -> None:
        self.queue = []

    def job_wrapper():
        pass

    def add_job_to_queue():
        pass

    def check_job_and_train():
        pass

    def get_job_info():
        pass


if __name__ == "__main__":
    lst = [TrainerNode(tr.ANNConfig())] * 3
    [i.fit() for i in lst]