from __future__ import annotations
import sys
from typing import Any
import _path_setup  # noqa: F401
import numpy as np
import torch
import torch.nn as nn
from mlp.backward import backprop
from mlp.forward import mlp_forward
from mlp.init import init_mlp
from torchmlp.model import ActivationName, MLP
from torchmlp.parity import compare_numpy_torch_grads, get_activation_map, load_numpy_weights

DEFAULT_ARCH = [2, 4, 1]
DEFAULT_SEED = 42
DEFAULT_ATOL = 1e-4

# gradient parity: PyTorch autograd vs NumPy backprop on one backward p
# returns a tuple of a boolean indicating whether the parity check passed and a list of dictionaries containing the results
# each dictionary contains the results for a layer
def run_parity_check(arch: list[int], activation: ActivationName, *, seed: int = DEFAULT_SEED, n: int = 8, atol: float = DEFAULT_ATOL) -> tuple[bool, list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, arch[0]))
    y = rng.normal(size=(n, 1))

    np.random.seed(seed)
    numpy_model = init_mlp(arch)
    act_forward, act_backward = get_activation_map()[activation]

    cache, pred = mlp_forward(numpy_model, x, activation=act_forward)
    numpy_grads = backprop(numpy_model, cache, y, pred, activation_backward=act_backward, lmbda=0.0)

    torch_model = MLP(arch, activation=activation)
    load_numpy_weights(torch_model, numpy_model)
    torch_model.train()

    x_t = torch.from_numpy(x).float()
    y_t = torch.from_numpy(y).float()
    criterion = nn.MSELoss()
    preds = torch_model(x_t)
    loss = criterion(preds, y_t)
    loss.backward()

    rows = compare_numpy_torch_grads(numpy_grads, torch_model, atol=atol)
    overall_row = rows[-1]
    return bool(overall_row["passed"]), rows[:-1]

# print the results for a layer
def _print_results(activation: ActivationName, layer_rows: list[dict[str, Any]]) -> bool:
    all_pass = True
    for row in layer_rows:
        layer = row["layer"]
        print(f"=== {activation} | {layer} ===")
        print(f"  weight max_abs_diff: {row['weight_max_abs_diff']:.6e}  allclose: {row['weight_allclose']}")
        print(f"  bias   max_abs_diff: {row['bias_max_abs_diff']:.6e}  allclose: {row['bias_allclose']}")
        all_pass = all_pass and row["passed"]
    return all_pass

# main function to run the parity check
def main() -> None:
    activations: list[ActivationName] = ["sigmoid", "tanh", "relu"]
    overall_pass = True

    print(f"Architecture: {DEFAULT_ARCH}  seed: {DEFAULT_SEED}  atol: {DEFAULT_ATOL}")
    print()

    for activation in activations:
        passed, layer_rows = run_parity_check(DEFAULT_ARCH, activation, seed=DEFAULT_SEED, atol=DEFAULT_ATOL)
        layer_pass = _print_results(activation, layer_rows)
        overall_pass = overall_pass and passed and layer_pass
        print()

    if overall_pass:
        print("OVERALL: PASS")
        sys.exit(0)

    print("OVERALL: FAIL")
    sys.exit(1)


if __name__ == "__main__":
    main()