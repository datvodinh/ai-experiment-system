import torch.nn as nn
from backend.configs import ANNConfig


class ANNModel(nn.Module):
    def __init__(
        self,
        config: ANNConfig
    ):
        super().__init__()
        self.config = config
        hidden_depth = [28*28, 1024, 512, 256, 128, 10]
        model = nn.ModuleList([])
        for i in range(len(hidden_depth)-1):
            model.append(nn.Linear(hidden_depth[i], hidden_depth[i+1]))
            if i != len(hidden_depth)-2:
                model.append(nn.Dropout(config.dropout))
        self.model = nn.Sequential(*model)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)


class CNNModel(nn.Module):
    def __init__(
        self,
        config: ANNConfig
    ):
        pass


# if __name__ == "__main__":
#     model = ANNModel(be.ANNConfig(hidden_size=[64,128,256,512,1024]))
#     print(model)
