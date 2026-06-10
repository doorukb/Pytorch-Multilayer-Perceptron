from __future__ import annotations
import mlflow
from torchmlp.config import TrainConfig

def is_mlflow_active() -> bool:
    return mlflow.active_run() is not None

def log_train_config(config: TrainConfig) -> None:
    if not is_mlflow_active():
        return
    mlflow.log_params(config.to_mlflow_params())

def log_metrics(metrics: dict[str, float], *, step: int | None, prefix: str = "") -> None:
    if not is_mlflow_active() or step is None:
        return
    for key, value in metrics.items():
        mlflow.log_metric(f"{prefix}{key}", value, step=step)