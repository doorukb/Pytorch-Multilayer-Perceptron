from __future__ import annotations
import itertools
from dataclasses import dataclass, replace
import _path_setup  # noqa: F401
import mlflow
import torch
from _mlflow_setup import configure_mlflow
from torchmlp.config import TrainConfig
from torchmlp.data import create_surface_split_dataloaders
from torchmlp.model import MLP
from torchmlp.tracking import REGISTERED_MODEL_NAME
from torchmlp.trainer import build_optimizer, evaluate, fit, resolve_criterion, resolve_device

MLFLOW_EXPERIMENT_NAME = "torchmlp-hyperparam-sweep"
WINNER_RUN_NAME = "sweep-winner"
SELECTION_CRITERION = "lowest validation MSE (min val_loss across epochs)"

BASE_SEED = 42
HIDDEN_SIZE_OPTIONS: list[list[int]] = [[5], [10], [5, 5]]
LEARNING_RATES: list[float] = [1e-2, 1e-3, 1e-4]
DROPOUT_VALUES: list[float] = [0.0, 0.1]

DEFAULT_ACTIVATION = "sigmoid"
DEFAULT_OPTIMIZER = "sgd"
DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 32
DEFAULT_N_SAMPLES = 1000

# hyperparameter grid sweep on the synthetic surface (Z = X² - Y² + 1.2 + noise)

# systematic experimentation: every grid configuration is trained and logged to MLflow.
# nothing is cherry-picked — the winner is selected by a fixed criterion before the test set is used.

# grid dimensions:
# - hidden_sizes (architecture): [5], [10], [5, 5]
# - learning_rate: 1e-2, 1e-3, 1e-4
# dropout: 0.0, 0.1

# selection criterion (stated before test evaluation): lowest validation MSE — min(val_loss) across training epochs

# reproducibility : BASE_SEED = 42 for data splits and weight initialization on every trial

# registry : sweep runs use registered_model_name=None to avoid flooding the model registry
# winner promotion is handled in experiments/04_model_promotion.py

# run from repo root : python experiments/03_hyperparam_sweep.py

# view results: mlflow ui

# open http://127.0.0.1:5000 and select experiment "torchmlp-hyperparam-sweep"

# MLflow 3+ file store: set MLFLOW_ALLOW_FILE_STORE=true before running, or use MLFLOW_TRACKING_URI=sqlite:///mlflow.db

@dataclass
class TrialResult:
    index: int
    config: TrainConfig
    history: dict[str, list[float]]
    best_val_loss: float
    run_id: str


def build_grid() -> list[TrainConfig]:
    configs: list[TrainConfig] = []

    for hidden_sizes, learning_rate, dropout in itertools.product(
        HIDDEN_SIZE_OPTIONS,
        LEARNING_RATES,
        DROPOUT_VALUES,
    ):
        configs.append(TrainConfig(
            task="regression",
            input_dim=2,
            output_dim=1,
            hidden_sizes=list(hidden_sizes),
            activation=DEFAULT_ACTIVATION,
            optimizer=DEFAULT_OPTIMIZER,
            learning_rate=learning_rate,
            dropout=dropout,
            epochs=DEFAULT_EPOCHS,
            batch_size=DEFAULT_BATCH_SIZE,
            n_samples=DEFAULT_N_SAMPLES,
            seed=BASE_SEED,
            registered_model_name=None,
        ))
    return configs

# return the winner configuration
def winner_config(config: TrainConfig) -> TrainConfig:
    return replace(config, registered_model_name=REGISTERED_MODEL_NAME)

# return the run name for the configuration
def run_name_for(config: TrainConfig) -> str:
    arch = "-".join(str(size) for size in config.hidden_sizes)
    return f"arch={arch}_lr={config.learning_rate:g}_do={config.dropout:g}"

# run the trial for the configuration
def run_trial(config: TrainConfig) -> tuple[dict[str, list[float]], float]:
    config.validate()
    torch.manual_seed(config.seed)
    train_loader, val_loader, _ = create_surface_split_dataloaders(n=config.n_samples, batch_size=config.batch_size, seed=config.seed)

    model = MLP(config.layer_sizes, activation=config.activation, dropout=config.dropout)
    optimizer = build_optimizer(model, config)
    history = fit(model, train_loader, val_loader, optimizer, config)
    best_val_loss = min(history["val_loss"])
    return history, best_val_loss

# run the trial with the test set
def run_with_test(config: TrainConfig) -> tuple[dict[str, list[float]], float, dict[str, float]]:
    config.validate()
    torch.manual_seed(config.seed)
    train_loader, val_loader, test_loader = create_surface_split_dataloaders(n=config.n_samples, batch_size=config.batch_size, seed=config.seed)

    model = MLP(config.layer_sizes, activation=config.activation, dropout=config.dropout)
    optimizer = build_optimizer(model, config)
    history = fit(model, train_loader, val_loader, optimizer, config)
    best_val_loss = min(history["val_loss"])

    device = resolve_device(config.device)
    criterion = resolve_criterion(config)
    test_metrics = evaluate(model, test_loader, criterion, device, task=config.task, step=config.epochs, metric_prefix="test_")
    return history, best_val_loss, test_metrics

# select the best configuration
def select_best(results: list[TrialResult]) -> TrialResult:
    return min(results, key=lambda result: (result.best_val_loss, result.index))

# format the architecture
def _format_arch(config: TrainConfig) -> str:
    return str(config.layer_sizes)

# print the ranked table
def print_ranked_table(results: list[TrialResult]) -> None:
    ranked = sorted(results, key=lambda result: (result.best_val_loss, result.index))
    print(f"{'rank':<5} {'best_val_mse':<14} {'architecture':<16} {'lr':<10} {'dropout':<8} run_id")
    print("-" * 80)
    for rank, result in enumerate(ranked, start=1):
        config = result.config
        print(
            f"{rank:<5} {result.best_val_loss:<14.6f} {_format_arch(config):<16} "
            f"{config.learning_rate:<10g} {config.dropout:<8g} {result.run_id}"
        )

def main() -> None:
    grid = build_grid()

    print("Hyperparameter sweep — synthetic surface (Z = X² - Y² + 1.2 + noise)")
    print(f"Selection criterion: {SELECTION_CRITERION}")
    print(f"Grid size: {len(grid)} configurations")
    print(f"Architectures: {HIDDEN_SIZE_OPTIONS}")
    print(f"Learning rates: {LEARNING_RATES}")
    print(f"Dropout values: {DROPOUT_VALUES}")
    print(f"Seed: {BASE_SEED}")
    print()
    print("Test set is not used during grid search or selection.")
    print()

    configure_mlflow(MLFLOW_EXPERIMENT_NAME)

    results: list[TrialResult] = []
    for index, config in enumerate(grid):
        with mlflow.start_run(run_name=run_name_for(config)):
            history, best_val_loss = run_trial(config)
            run_id = mlflow.active_run().info.run_id

        results.append(TrialResult(
            index=index,
            config=config,
            history=history,
            best_val_loss=best_val_loss,
            run_id=run_id,
        ))

    best = select_best(results)

    print("Ranked results (all trials):")
    print_ranked_table(results)
    print()
    print("Winner selected by validation MSE (not test loss).")
    print("Re-training winner for single held-out test evaluation and model registry...")
    print(f"  architecture={best.config.layer_sizes}  lr={best.config.learning_rate:g}  dropout={best.config.dropout:g}")
    print(f"  best val MSE (selection)={best.best_val_loss:.6f}  grid run_id={best.run_id}")
    print()

    with mlflow.start_run(run_name=WINNER_RUN_NAME):
        winner = winner_config(best.config)
        _, winner_val_loss, test_metrics = run_with_test(winner)
        winner_run_id = mlflow.active_run().info.run_id

    test_loss = test_metrics["loss"]

    print(f"Winner test loss (MSE): {test_loss:.6f}")
    print(f"Winner best val MSE:    {winner_val_loss:.6f}")
    print()
    print(f"Best grid trial run:    {best.run_id}")
    print(f"Winner registration run: {winner_run_id}")
    print(f"Registered model:       {REGISTERED_MODEL_NAME} (new version created)")
    print()
    print(f"MLflow experiment: {MLFLOW_EXPERIMENT_NAME}")
    print("View all trials: mlflow ui  (then open http://127.0.0.1:5000)")

if __name__ == "__main__":
    main()