import os
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

NUMPY_MLP_REPO_URL = "https://github.com/doorukb/Multilayer-perceptron.git"

# resolve the path to the numpy mlp src
def _resolve_numpy_mlp_src() -> Path | None:
    env_path = os.environ.get("NUMPY_MLP_SRC")
    if env_path:
        path = Path(env_path)
        return path if path.is_dir() else None

    sibling = Path(__file__).resolve().parents[2] / "Multilayer-Perceptron" / "src"
    return sibling if sibling.is_dir() else None



_numpy_src = _resolve_numpy_mlp_src()
if _numpy_src is not None:
    numpy_src_str = str(_numpy_src)
    if numpy_src_str not in sys.path:
        sys.path.insert(0, numpy_src_str)

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))