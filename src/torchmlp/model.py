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

def _init_linear_xavier(linear: nn.Linear, fan_in: int) -> None:
    std = math.sqrt(1.0 / fan_in)
    nn.init.normal_(linear.weight, mean=0.0, std=std)
    nn.init.zeros_(linear.bias)

# configurable multilayer perceptron for scalar regression

# NumPy reference stores weights in explicit dicts (model["W0"], ...) and
# updates them manually in a gradient-descent loop

# nn.Module registers parameters automatically, exposes state_dict(), and pairs with
# optimizer.step() so gradients from loss.backward() drive updates
class MLP(nn.Module):

    def __init__(self, layer_sizes: Sequence[int], activation: ActivationName = "sigmoid") -> None:
        super().__init__()
        sizes = list(layer_sizes)

        if len(sizes) < 2:
            raise ValueError("layer_sizes must contain at least input and output dimensions")

        self.layer_sizes = sizes
        self.activation_name = activation
        self.activation = make_activation(activation)
        self.linears = nn.ModuleList(nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1))

        for i, linear in enumerate(self.linears):
            _init_linear_xavier(linear, fan_in=sizes[i])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, linear in enumerate(self.linears):
            x = linear(x)
            if i < len(self.linears) - 1:
                x = self.activation(x)
        return x