import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from torchmlp.config import TrainConfig
from torchmlp.data import create_surface_dataloaders
from torchmlp.model import MLP
from torchmlp.trainer import evaluate, fit, resolve_device, train, train_one_epoch


def _fast_config(**overrides) -> TrainConfig:
    defaults = {
        "n_samples": 200,
        "batch_size": 32,
        "epochs": 20,
        "seed": 42,
        "learning_rate": 1e-3,
    }
    defaults.update(overrides)
    return TrainConfig(**defaults)


def _make_training_setup(config: TrainConfig | None = None):
    config = config or _fast_config()
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
    device = resolve_device(config.device)
    return model, train_loader, val_loader, optimizer, criterion, device, config


def _classification_loader(batch_size: int = 16, seed: int = 0) -> DataLoader:
    gen = torch.Generator().manual_seed(seed)
    features = torch.randn(64, 4, generator=gen)
    labels = torch.randint(0, 2, (64,), generator=gen)
    return DataLoader(TensorDataset(features, labels), batch_size=batch_size, shuffle=False)


def test_fit_reduces_train_loss():
    model, train_loader, val_loader, optimizer, criterion, device, config = _make_training_setup()
    history = fit(model, train_loader, val_loader, optimizer, config, criterion=criterion)
    assert len(history["train_loss"]) == config.epochs
    assert history["train_loss"][-1] < history["train_loss"][0]


def test_train_from_config_reduces_loss():
    config = _fast_config(epochs=15)
    history = train(config)
    assert len(history["train_loss"]) == 15
    assert history["train_loss"][-1] < history["train_loss"][0]


def test_eval_runs_in_eval_mode():
    model, train_loader, val_loader, optimizer, criterion, device, config = _make_training_setup()
    model.to(device)
    metrics = evaluate(model, val_loader, criterion, device, task=config.task)
    assert model.training is False
    assert "loss" in metrics


def test_train_runs_in_train_mode():
    model, train_loader, _, optimizer, criterion, device, config = _make_training_setup()
    model.to(device)
    model.eval()
    train_one_epoch(model, train_loader, optimizer, criterion, device, task=config.task)
    assert model.training is True


def test_zero_grad_prevents_accumulation():
    model = MLP([2, 3, 1])
    x = torch.randn(4, 2)
    y = torch.randn(4, 1)
    criterion = nn.MSELoss()
    param = next(model.parameters())

    model.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    single_grad = param.grad.detach().clone()

    model.zero_grad()
    criterion(model(x), y).backward()
    criterion(model(x), y).backward()
    accumulated_grad = param.grad.detach().clone()

    model.zero_grad()
    criterion(model(x), y).backward()
    model.zero_grad()
    criterion(model(x), y).backward()
    cleared_grad = param.grad.detach().clone()

    assert torch.allclose(accumulated_grad, single_grad * 2)
    assert torch.allclose(cleared_grad, single_grad)


def test_resolve_device_defaults_to_cpu_when_no_cuda():
    device = resolve_device(None)
    if torch.cuda.is_available():
        assert device.type == "cuda"
    else:
        assert device.type == "cpu"
    assert resolve_device("cpu").type == "cpu"


def test_fit_reproducible_with_seed():
    def run_fit(seed: int):
        config = _fast_config(epochs=10, seed=seed)
        model, train_loader, val_loader, optimizer, criterion, device, _ = _make_training_setup(config)
        return fit(model, train_loader, val_loader, optimizer, config, criterion=criterion)

    history_a = run_fit(42)
    history_b = run_fit(42)
    assert history_a["train_loss"] == history_b["train_loss"]
    assert history_a["val_loss"] == history_b["val_loss"]


def test_classification_evaluate_returns_metrics_dict():
    torch.manual_seed(0)
    model = MLP([4, 8, 2])
    loader = _classification_loader()
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cpu")
    metrics = evaluate(model, loader, criterion, device, task="classification")
    assert set(metrics) == {"loss", "accuracy", "f1", "auc"}
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
    assert 0.0 <= metrics["auc"] <= 1.0


def test_classification_train_epoch_runs():
    torch.manual_seed(0)
    model = MLP([4, 8, 2])
    loader = _classification_loader()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cpu")
    loss = train_one_epoch(model, loader, optimizer, criterion, device, task="classification")
    assert loss >= 0.0
