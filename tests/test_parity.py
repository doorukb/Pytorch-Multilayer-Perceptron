import sys
from pathlib import Path
import pytest
from importlib import import_module # the parity check module

# check if the numpy mlp src is available
_numpy_src = Path(__file__).resolve().parents[2] / "Multilayer-Perceptron" / "src"

if not _numpy_src.is_dir() and "NUMPY_MLP_SRC" not in __import__("os").environ:
    pytest.skip("NumPy mlp reference not available (clone Multilayer-Perceptron sibling or set NUMPY_MLP_SRC)", allow_module_level=True)

_parity_mod = import_module("01_parity_check")
run_parity_check = _parity_mod.run_parity_check

# test the parity of the gradients between the numpy and pytorch implementations
@pytest.mark.parametrize("activation", ["sigmoid", "tanh", "relu"])
def test_parity_gradients_match_numpy(activation):
    passed, _ = run_parity_check([2, 4, 1], activation, seed=42)
    assert passed