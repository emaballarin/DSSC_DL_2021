from typing import List, Union
from torch import nn, Tensor
import torch.nn.functional as F


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
