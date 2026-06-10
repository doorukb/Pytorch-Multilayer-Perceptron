from __future__ import annotations
import os
import sys
from pathlib import Path

NUMPY_MLP_REPO_URL = "https://github.com/doorukb/Multilayer-perceptron.git"

# adds torchmlp and NumPy mlp reference to sys.path for experiments
# the NumPy mlp reference is a repository that contains the code for the NumPy implementation of the multilayer perceptron
def resolve_numpy_mlp_src() -> Path:
    env_path = os.environ.get("NUMPY_MLP_SRC")
    if env_path:
        return Path(env_path)

    sibling = Path(__file__).resolve().parents[2] / "Multilayer-Perceptron" / "src"
    if sibling.is_dir():
        return sibling

    raise FileNotFoundError(
        "NumPy mlp reference not found. Clone "
        f"{NUMPY_MLP_REPO_URL} as a sibling Multilayer-Perceptron repo, "
        "or set NUMPY_MLP_SRC to its src/ directory."
    )

# setup the paths for the experiments
def setup_paths() -> None:
    project_src = Path(__file__).resolve().parents[1] / "src"
    numpy_src = resolve_numpy_mlp_src()

    for path in (project_src, numpy_src):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

setup_paths()