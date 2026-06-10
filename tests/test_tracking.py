import mlflow
import torch
import torch.nn as nn
from mlflow.tracking import MlflowClient
from torch.utils.data import DataLoader, TensorDataset

from torchmlp.config import TrainConfig
from torchmlp.data import create_surface_dataloaders
from torchmlp.model import MLP
from torchmlp.tracking import is_mlflow_active, log_metrics, log_train_config
from torchmlp.trainer import fit


def _configure_tracking(tmp_path) -> None:
    db_path = (tmp_path / "mlflow.db").as_posix()
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    mlflow.set_experiment("test-tracking")


def _fast_config(**overrides) -> TrainConfig:
    defaults = {
        "n_samples": 200,
        "batch_size": 32,
        "epochs": 3,
        "seed": 42,
        "learning_rate": 1e-3,
    }
    defaults.update(overrides)
    return TrainConfig(**defaults)


def _make_fit_setup(config: TrainConfig):
    torch.manual_seed(config.seed)
    model = MLP(config.layer_sizes, activation=config.activation, dropout=config.dropout)
    train_loader, val_loader = create_surface_dataloaders(
        train_size=100,
        test_size=20,
        batch_size=config.batch_size,
        seed=config.seed,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()
    return model, train_loader, val_loader, optimizer, criterion


def _classification_loaders(batch_size: int = 16, seed: int = 0) -> tuple[DataLoader, DataLoader]:
    gen = torch.Generator().manual_seed(seed)
    features = torch.randn(64, 4, generator=gen)
    labels = torch.randint(0, 2, (64,), generator=gen)
    dataset = TensorDataset(features, labels)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=gen)
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def test_log_metrics_noop_without_active_run():
    assert is_mlflow_active() is False
    log_metrics({"loss": 1.0}, step=0, prefix="train_")


def test_log_train_config_logs_all_params(tmp_path):
    _configure_tracking(tmp_path)
    config = _fast_config(hidden_sizes=[5, 10], device=None)

    with mlflow.start_run():
        log_train_config(config)
        run_id = mlflow.active_run().info.run_id

    logged = MlflowClient().get_run(run_id).data.params
    assert logged == config.to_mlflow_params()


def test_fit_logs_per_epoch_curves(tmp_path):
    _configure_tracking(tmp_path)
    config = _fast_config(epochs=3)
    model, train_loader, val_loader, optimizer, criterion = _make_fit_setup(config)

    with mlflow.start_run():
        fit(model, train_loader, val_loader, optimizer, config, criterion=criterion)
        run_id = mlflow.active_run().info.run_id

    client = MlflowClient()
    train_history = client.get_metric_history(run_id, "train_loss")
    val_history = client.get_metric_history(run_id, "val_loss")

    assert len(train_history) == config.epochs
    assert len(val_history) == config.epochs
    assert {point.step for point in train_history} == {0, 1, 2}
    assert {point.step for point in val_history} == {0, 1, 2}


def test_fit_logs_classification_val_metrics(tmp_path):
    _configure_tracking(tmp_path)
    config = TrainConfig(
        task="classification",
        input_dim=4,
        output_dim=2,
        hidden_sizes=[8],
        epochs=3,
        batch_size=16,
        seed=0,
    )
    torch.manual_seed(config.seed)
    model = MLP(config.layer_sizes, activation=config.activation, dropout=config.dropout)
    train_loader, val_loader = _classification_loaders(batch_size=config.batch_size, seed=config.seed)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    with mlflow.start_run():
        fit(model, train_loader, val_loader, optimizer, config)
        run_id = mlflow.active_run().info.run_id

    client = MlflowClient()
    for metric_name in ("val_accuracy", "val_f1", "val_auc"):
        history = client.get_metric_history(run_id, metric_name)
        assert len(history) == config.epochs
