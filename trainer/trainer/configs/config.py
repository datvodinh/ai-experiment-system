from pydantic import BaseModel


class ANNConfig(BaseModel):
    max_epochs: int = 50
    batch_size: int = 32
    lr: float = 1e-4
    dropout: float = 0.1
