from __future__ import annotations
import os
import sys
from pathlib import Path

NUMPY_MLP_REPO_URL = "https://github.com/doorukb/Multilayer-perceptron.git"

# locate the NumPy mlp reference src/ directory when available
def find_numpy_mlp_src() -> Path | None:
    env_path = os.environ.get("NUMPY_MLP_SRC")
    if env_path:
        return Path(env_path)

    sibling = Path(__file__).resolve().parents[2] / "Multilayer-Perceptron" / "src"
    if sibling.is_dir():
        return sibling

    return None

# resolve the NumPy mlp reference src/ directory or raise with setup instructions
def resolve_numpy_mlp_src() -> Path:
    numpy_src = find_numpy_mlp_src()
    if numpy_src is not None:
        return numpy_src

    raise FileNotFoundError(
        "NumPy mlp reference not found. Clone "
        f"{NUMPY_MLP_REPO_URL} as a sibling Multilayer-Perceptron repo, "
        "or set NUMPY_MLP_SRC to its src/ directory."
    )

# setup the paths for the experiments
def setup_paths() -> None:
    project_src = Path(__file__).resolve().parents[1] / "src"
    paths = [project_src]

    numpy_src = find_numpy_mlp_src()
    if numpy_src is not None:
        paths.append(numpy_src)

    for path in paths:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

setup_paths()
