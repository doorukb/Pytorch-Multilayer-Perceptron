from __future__ import annotations
from collections.abc import Callable
from typing import Any
import numpy as np
import torch
from torchmlp.model import ActivationName, MLP

# get the activation map for the activations
# returns a dictionary with the activation name as the key and the tuple of the forward and backward functions as the value
def get_activation_map() -> dict[ActivationName, tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]]:
    from mlp.activations import (
        relu_backward,
        relu_forward,
        sigmoid_backward,
        sigmoid_forward,
        tanh_backward,
        tanh_forward,
    )
    return {
        "sigmoid": (sigmoid_forward, sigmoid_backward),
        "tanh": (tanh_forward, tanh_backward),
        "relu": (relu_forward, relu_backward),
    }

# load the numpy weights into the model
# copy NumPy W{i} matrices into PyTorch nn.Linear layers
# NumPy W: (in_dim + 1, out_dim) with bias as last row
# PyTorch: weight (out_dim, in_dim), bias (out_dim,)
def load_numpy_weights(model: MLP, numpy_model: dict[str, np.ndarray]) -> None:
    for i, linear in enumerate(model.linears):
        weight_np = numpy_model[f"W{i}"]
        linear.weight.data.copy_(torch.from_numpy(weight_np[:-1, :].T).float())
        linear.bias.data.copy_(torch.from_numpy(weight_np[-1, :]).float())

# compare the numpy gradients to the pytorch gradients
def compare_numpy_torch_grads(numpy_grads: dict[str, np.ndarray], model: MLP, *, atol: float = 1e-4) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    all_pass = True

    for i, linear in enumerate(model.linears):
        dW = numpy_grads[f"dW{i}"]
        weight_np = dW[:-1, :].T
        bias_np = dW[-1, :]

        torch_weight = linear.weight.grad.detach().cpu().numpy()
        torch_bias = linear.bias.grad.detach().cpu().numpy()

        weight_diff = float(np.max(np.abs(weight_np - torch_weight)))
        bias_diff = float(np.max(np.abs(bias_np - torch_bias)))
        weight_ok = bool(np.allclose(weight_np, torch_weight, atol=atol))
        bias_ok = bool(np.allclose(bias_np, torch_bias, atol=atol))
        layer_ok = weight_ok and bias_ok
        all_pass = all_pass and layer_ok

        rows.append({
            "layer": f"W{i}",
            "weight_max_abs_diff": weight_diff,
            "weight_allclose": weight_ok,
            "bias_max_abs_diff": bias_diff,
            "bias_allclose": bias_ok,
            "passed": layer_ok,
        })

    rows.append({"passed": all_pass, "overall": True})
    return rows