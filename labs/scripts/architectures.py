import torch
from torch import nn

class MLP(nn.Module):
   def __init__(width_layers, size_in=784, num_classes=10, activ_hidden=nn.ReLU, activ_out=None)
