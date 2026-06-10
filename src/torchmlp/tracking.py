from __future__ import annotations
import os
import tempfile
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import torch.nn as nn
from torchmlp.config import TrainConfig

matplotlib.use("Agg")

REGISTERED_MODEL_NAME = "torchmlp-mlp"

# check if the mlflow is active
def is_mlflow_active() -> bool:
    return mlflow.active_run() is not None

# log the train configuration
def log_train_config(config: TrainConfig) -> None:
    if not is_mlflow_active():
        return
    mlflow.log_params(config.to_mlflow_params())

# log the metrics
def log_metrics(metrics: dict[str, float], *, step: int | None, prefix: str = "") -> None:
    if not is_mlflow_active() or step is None:
        return
    for key, value in metrics.items():
        mlflow.log_metric(f"{prefix}{key}", value, step=step)

# save the learning curve
def save_learning_curve(history: dict[str, list[float]], path: str | Path) -> None:
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    epochs = range(1, len(train_loss) + 1)

    fig, ax = plt.subplots()
    ax.plot(epochs, train_loss, label="train loss")
    ax.plot(epochs, val_loss, label="val loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("learning curve")
    ax.legend()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

# log the learning curve
def log_learning_curve(history: dict[str, list[float]], *, artifact_path: str = "plots/learning_curve.png") -> None:
    if not is_mlflow_active():
        return

    artifact_dir = str(Path(artifact_path).parent)
    artifact_name = Path(artifact_path).name
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir) / artifact_name

    try:
        save_learning_curve(history, temp_path)
        mlflow.log_artifact(str(temp_path), artifact_path=artifact_dir or None)
    finally:
        temp_path.unlink(missing_ok=True)
        os.rmdir(temp_dir)

# log the pytorch model
def log_pytorch_model(model: nn.Module, *, artifact_path: str = "model", registered_model_name: str | None = REGISTERED_MODEL_NAME) -> None:
    if not is_mlflow_active():
        return

    # log the model on the cpu
    original_device = next(model.parameters()).device
    model.cpu()
    try:
        log_kwargs: dict[str, str] = {"artifact_path": artifact_path}
        if registered_model_name is not None:
            log_kwargs["registered_model_name"] = registered_model_name
        mlflow.pytorch.log_model(model, **log_kwargs)
    finally:
        model.to(original_device)