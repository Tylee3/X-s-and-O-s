"""
models.py -- the two trained classifiers.

MLP  (Step 4): flattens the 32x32 image to 1024 numbers and pushes them through fully
               connected layers.  It has no idea which inputs are neighbours: pixel (5,5)
               and pixel (5,6) are just input #165 and input #166 to it.
CNN  (Step 5): added in Step 5.
"""
import torch
import torch.nn as nn


class MLP(nn.Module):
    """1024 -> hidden[0] -> hidden[1] -> 1 logit.

    Linear   : every output unit = weighted sum of ALL its inputs + bias  (a perceptron each)
    ReLU     : max(0, x).  Without a non-linearity between layers, stacking linear layers
               collapses to ONE linear layer, i.e. a single perceptron -- exactly the
               model that could not separate the flat ovals in Step 3.
    Dropout  : during training, randomly zero a fraction of the units each step so no
               single unit can be relied on; a cheap way to fight memorisation.
    Output   : one number (a "logit").  >0 means X, <0 means O.  sigmoid(logit) = P(X).
    """

    def __init__(self, size: int = 32, hidden=(128, 64), dropout: float = 0.3):
        super().__init__()
        layers, n_in = [], size * size
        for n_out in hidden:
            layers += [nn.Linear(n_in, n_out), nn.ReLU(), nn.Dropout(dropout)]
            n_in = n_out
        layers.append(nn.Linear(n_in, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):                       # x: [batch, 1, size, size]
        return self.net(x.flatten(1)).squeeze(1)  # -> [batch] logits


def build(name: str, size: int, **kw) -> nn.Module:
    if name == "mlp":
        return MLP(size=size, **kw)
    if name == "cnn":
        from models_cnn import CNN               # Step 5
        return CNN(size=size, **kw)
    raise ValueError(name)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
