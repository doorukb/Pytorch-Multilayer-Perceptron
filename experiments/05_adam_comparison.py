from __future__ import annotations
from dataclasses import dataclass
import _path_setup  # noqa: F401
import mlflow
import torch
from _mlflow_setup import configure_mlflow
from torchmlp.config import TrainConfig
from torchmlp.data import create_surface_split_dataloaders
from torchmlp.model import MLP
from torchmlp.trainer import build_optimizer, evaluate, fit, resolve_criterion, resolve_device

MLFLOW_EXPERIMENT_NAME = "torchmlp-sgd-vs-adam"

HIDDEN_SIZES = [10]
DEFAULT_ACTIVATION = "sigmoid"
DEFAULT_BATCH_SIZE = 32
N_SAMPLES = 1000
SEED = 42
EPOCHS = 200

SGD_LEARNING_RATE = 0.05
ADAM_LEARNING_RATE = 1e-3

# compare SGD (lr=0.05) vs Adam (lr=1e-3) on the NumPy-winning regression architecture [2, 10, 1]
# 200 epochs x ~31 mini-batch steps per epoch gives Adam enough room to show faster convergence

# run from repo root: python experiments/05_adam_comparison.py

# view results in the MLflow UI : mlflow ui
# open http://127.0.0.1:5000 and select experiment "torchmlp-sgd-vs-adam"

# MLflow 3+ file store: set MLFLOW_ALLOW_FILE_STORE=true before running, or use
# MLFLOW_TRACKING_URI=sqlite:///mlflow.db

@dataclass
class RunResult:
    method: str
    config: TrainConfig
    history: dict[str, list[float]]
    test_metrics: dict[str, float]
    run_id: str


def build_configs() -> list[tuple[str, str, TrainConfig]]:
    shared = dict(
        task="regression",
        input_dim=2,
        output_dim=1,
        hidden_sizes=HIDDEN_SIZES,
        activation=DEFAULT_ACTIVATION,
        batch_size=DEFAULT_BATCH_SIZE,
        n_samples=N_SAMPLES,
        seed=SEED,
        epochs=EPOCHS,
        registered_model_name=None,
    )
    return [
        ("SGD", "sgd-lr0.05", TrainConfig(optimizer="sgd", learning_rate=SGD_LEARNING_RATE, **shared)),
        ("Adam", "adam-lr0.001", TrainConfig(optimizer="adam", learning_rate=ADAM_LEARNING_RATE, **shared)),
    ]


def run_comparison(config: TrainConfig) -> tuple[dict[str, list[float]], dict[str, float]]:
    config.validate()
    torch.manual_seed(config.seed)
    train_loader, val_loader, test_loader = create_surface_split_dataloaders(
        n=config.n_samples, batch_size=config.batch_size, seed=config.seed,
    )

    model = MLP(config.layer_sizes, activation=config.activation, dropout=config.dropout)
    optimizer = build_optimizer(model, config)
    history = fit(model, train_loader, val_loader, optimizer, config)

    device = resolve_device(config.device)
    criterion = resolve_criterion(config)
    test_metrics = evaluate(
        model, test_loader, criterion, device,
        task=config.task, step=config.epochs, metric_prefix="test_",
    )
    return history, test_metrics


def print_comparison_table(results: list[RunResult]) -> None:
    print(
        f"{'method':<8} {'optimizer':<10} {'lr':<10} {'epochs':<8} "
        f"{'final_train_mse':<18} {'final_val_mse':<16} {'test_mse':<12}"
    )
    print("-" * 90)
    for result in results:
        config = result.config
        print(
            f"{result.method:<8} {config.optimizer:<10} {config.learning_rate:<10g} {config.epochs:<8} "
            f"{result.history['train_loss'][-1]:<18.6f} {result.history['val_loss'][-1]:<16.6f} "
            f"{result.test_metrics['loss']:<12.6f}"
        )

def main() -> None:
    configure_mlflow(MLFLOW_EXPERIMENT_NAME)

    print("SGD vs Adam — synthetic surface regression (Z = X² - Y² + 1.2 + noise)")
    print(f"Architecture: [2, 10, 1]  activation: {DEFAULT_ACTIVATION}")
    print(f"n_samples: {N_SAMPLES}  seed: {SEED}  epochs: {EPOCHS}  batch_size: {DEFAULT_BATCH_SIZE}")
    print(f"SGD:  lr={SGD_LEARNING_RATE}")
    print(f"Adam: lr={ADAM_LEARNING_RATE:g}")
    print()

    results: list[RunResult] = []
    for method, run_name, config in build_configs():
        with mlflow.start_run(run_name=run_name):
            history, test_metrics = run_comparison(config)
            run_id = mlflow.active_run().info.run_id
        results.append(RunResult(method=method, config=config, history=history, test_metrics=test_metrics, run_id=run_id))

    print("Comparison (final epoch metrics):")
    print_comparison_table(results)
    print()
    for result in results:
        print(f"  {result.method} run: {result.run_id}")
    print()
    print(f"MLflow experiment: {MLFLOW_EXPERIMENT_NAME}")
    print("View results: mlflow ui  (then open http://127.0.0.1:5000)")

if __name__ == "__main__":
    main()