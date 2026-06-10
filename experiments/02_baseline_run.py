from __future__ import annotations
import _path_setup  # noqa: F401
import mlflow
import torch
from _mlflow_setup import configure_mlflow
from torchmlp.config import TrainConfig
from torchmlp.data import create_surface_split_dataloaders
from torchmlp.model import MLP
from torchmlp.trainer import build_optimizer, evaluate, fit, resolve_criterion, resolve_device

MLFLOW_EXPERIMENT_NAME = "torchmlp-surface"
DEFAULT_RUN_NAME = "baseline-surface"

DEFAULT_HIDDEN_SIZES = [5]
DEFAULT_ACTIVATION = "sigmoid"
DEFAULT_OPTIMIZER = "sgd"
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 32
DEFAULT_N_SAMPLES = 1000
DEFAULT_SEED = 42

# baseline training run on the synthetic surface (Z = X² - Y² + 1.2 + noise)

# experiment : fixed seed (42), TrainConfig defaults, train/val/test metrics logged to MLflow when run inside mlflow.start_run()

# run from repo root: python experiments/02_baseline_run.py

# view results in the MLflow UI : mlflow ui
# open http://127.0.0.1:5000 and select experiment "torchmlp-surface"

# MLflow 3+ file store: set MLFLOW_ALLOW_FILE_STORE=true before running, or use
# MLFLOW_TRACKING_URI=sqlite:///mlflow.db

def run_baseline(config: TrainConfig) -> tuple[dict[str, list[float]], dict[str, float]]:
    config.validate()
    torch.manual_seed(config.seed)
    train_loader, val_loader, test_loader = create_surface_split_dataloaders(n=config.n_samples, batch_size=config.batch_size, seed=config.seed)

    model = MLP(config.layer_sizes, activation=config.activation, dropout=config.dropout)
    optimizer = build_optimizer(model, config)
    history = fit(model, train_loader, val_loader, optimizer, config)

    device = resolve_device(config.device)
    criterion = resolve_criterion(config)
    test_metrics = evaluate(model, test_loader, criterion, device, task=config.task, step=config.epochs, metric_prefix="test_")

    return history, test_metrics

def main() -> None:
    configure_mlflow(MLFLOW_EXPERIMENT_NAME)

    config = TrainConfig(
        task="regression",
        input_dim=2,
        output_dim=1,
        hidden_sizes=DEFAULT_HIDDEN_SIZES,
        activation=DEFAULT_ACTIVATION,
        optimizer=DEFAULT_OPTIMIZER,
        learning_rate=DEFAULT_LEARNING_RATE,
        epochs=DEFAULT_EPOCHS,
        batch_size=DEFAULT_BATCH_SIZE,
        n_samples=DEFAULT_N_SAMPLES,
        seed=DEFAULT_SEED,
    )

    print("Baseline run — synthetic surface (Z = X² - Y² + 1.2 + noise)")
    print(f"Architecture: {config.layer_sizes}  activation: {config.activation}")
    print(f"optimizer: {config.optimizer}  lr: {config.learning_rate}  epochs: {config.epochs}  batch_size: {config.batch_size}")
    print(f"n_samples: {config.n_samples}  seed: {config.seed}")
    print()

    with mlflow.start_run(run_name=DEFAULT_RUN_NAME):
        history, test_metrics = run_baseline(config)
        run_id = mlflow.active_run().info.run_id

    final_train_loss = history["train_loss"][-1]
    final_val_loss = history["val_loss"][-1]
    test_loss = test_metrics["loss"]

    print(f"Final train loss (MSE): {final_train_loss:.6f}")
    print(f"Final val loss (MSE):   {final_val_loss:.6f}")
    print(f"Test loss (MSE):        {test_loss:.6f}")
    print()
    print(f"MLflow run: {run_id}  experiment: {MLFLOW_EXPERIMENT_NAME}")
    print("View results: mlflow ui  (then open http://127.0.0.1:5000)")


if __name__ == "__main__":
    main()