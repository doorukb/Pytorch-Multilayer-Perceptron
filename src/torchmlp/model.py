from __future__ import annotations
import math
from collections.abc import Sequence
from typing import Literal
import torch
import torch.nn as nn

ActivationName = Literal["sigmoid", "tanh", "relu"]

def make_activation(name: ActivationName) -> nn.Module:
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "tanh":
        return nn.Tanh()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"Unsupported activation: {name!r}")

# initialize the linear layer with xavier initialization
def _init_linear_xavier(linear: nn.Linear, fan_in: int) -> None:
    std = math.sqrt(1.0 / fan_in)
    nn.init.normal_(linear.weight, mean=0.0, std=std)
    nn.init.zeros_(linear.bias)

# the multilayer perceptron model we will use to train the model
class MLP(nn.Module):
    def __init__(self, layer_sizes: Sequence[int], activation: ActivationName = "sigmoid", dropout: float = 0.0) -> None:
        super().__init__()
        sizes = list(layer_sizes)

        if len(sizes) < 2:
            raise ValueError("layer_sizes must contain at least input and output dimensions")

        self.layer_sizes = sizes
        self.activation_name = activation
        self.dropout_p = dropout
        self.activation = make_activation(activation)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else None
        self.linears = nn.ModuleList(nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1))

        for i, linear in enumerate(self.linears):
            _init_linear_xavier(linear, fan_in=sizes[i])

    # return logits
    # use predict_proba at inference for class probabilities
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, linear in enumerate(self.linears):
            x = linear(x)
            if i < len(self.linears) - 1:
                x = self.activation(x)

                if self.dropout is not None:
                    x = self.dropout(x)
        return x

    # softmax probabilities from logits (not used during CrossEntropy training)
    def predict_proba(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits, dim=-1)