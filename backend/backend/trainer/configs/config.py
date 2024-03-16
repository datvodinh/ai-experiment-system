from pydantic import BaseModel


class ANNConfig(BaseModel):
    max_epochs: int = 20
    batch_size: int = 256
    lr: float = 1e-3
    dropout: float = 0.2


