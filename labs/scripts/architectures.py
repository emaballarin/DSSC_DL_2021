from typing import List, Union
from torch import nn, Tensor
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 16),
            nn.ReLU(),

            nn.BatchNorm1d(num_features=16),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Dropout(p=.2), # we add a dropout here. it's referred to the previous layer (with 32 neurons)

            nn.BatchNorm1d(num_features=32),
            nn.Linear(32, 24),
            nn.ReLU(),

            nn.BatchNorm1d(num_features=24),
            nn.Linear(24, 10)
        )


    def forward(self, X):
        return self.layers(X)


# Parameterized SoftMax Classifier (logistic regression as a NN)
class SMC(nn.Module):
    def __init__(self, fin: int, fout: int) -> None:
        super().__init__()
        self.layer1: nn.Module = nn.Linear(in_features=fin, out_features=fout)

    def forward(self, x: Tensor):
        out: Tensor = self.layer1(x)
        out: Tensor = F.log_softmax(out, dim=1)
        return out


# Fully-Connected Block
# Use ebtorch instead!
