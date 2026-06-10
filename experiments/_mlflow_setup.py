from __future__ import annotations
import os
from pathlib import Path
import mlflow

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# configure the mlflow tracking
def configure_mlflow(experiment_name: str) -> None:
    if uri := os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(uri)
    else:
        mlruns_path = (PROJECT_ROOT / "mlruns").as_posix()
        mlflow.set_tracking_uri(f"file:///{mlruns_path}")
    mlflow.set_experiment(experiment_name)