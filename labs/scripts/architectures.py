from torch import nn, Tensor
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 16),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=16),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Dropout(
                p=0.2
            ),  # we add a dropout here. it's referred to the previous layer (with 32 neurons)
            nn.BatchNorm1d(num_features=32),
            nn.Linear(32, 24),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=24),
            nn.Linear(24, 10),
        )

    def forward(self, xin):
        return self.layers(xin)


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


class MLPCustom(nn.Module):
    def __init__(self, layers_list: list, flatten_input=True):
        super().__init__()
        layers = self._build_layers(layers_list)
        if flatten_input:
            layers.insert(0, nn.Flatten())
        self.layers = nn.Sequential(*layers)

    def _build_layers(self, layers_list: list):
        layers = []
        prev_in = None
        for i, l_dict in enumerate(layers_list):
            if not isinstance(l_dict, dict):
                raise RuntimeError(
                    f"layers_list must be a list of dicts. Entry {i}, found {type(l_dict)}"
                )
            if "n_in" not in l_dict.keys():
                if i == 0:
                    raise RuntimeError(
                        "n_in must be specified inside the dict of the first item of layers_list"
                    )
                l_dict["n_in"] = prev_in

            layers.extend(self._build_single_layer(**l_dict))
            prev_in = l_dict["n_out"]
        return layers

    def _build_single_layer(
        self,
        n_in: int,
        n_out: int,
        activ=nn.ReLU,
        batchnorm=True,
        dropout_p=None,
        bias=True,
    ):
        layers = []
        if batchnorm:
            layers.append(nn.BatchNorm1d(n_in))
        layers.append(nn.Linear(n_in, n_out, bias=bias))
        if activ is not None:
            layers.append(activ())
        if dropout_p is not None and (dropout_p > 0.0 and dropout_p < 1.0):
            layers.append(nn.Dropout(p=dropout_p))
        return layers

    def forward(self, xin: Tensor):
        return self.layers(xin)
