from __future__ import annotations
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmlp.config import TrainConfig
from torchmlp.data import create_surface_split_dataloaders
from torchmlp.metrics import compute_classification_metrics
from torchmlp.model import MLP
from torchmlp.tracking import is_mlflow_active, log_learning_curve, log_metrics, log_pytorch_model, log_train_config

# resolve the device to use for the training
def resolve_device(device: str | torch.device | None = None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)

# resolve the criterion to use for the training
def resolve_criterion(config: TrainConfig) -> nn.Module:
    if config.task == "regression":
        return nn.MSELoss()
    if config.task == "classification":
        return nn.CrossEntropyLoss()
    raise ValueError(f"Unsupported task: {config.task!r}")

# prepare the targets for the training
def _prepare_targets(targets: torch.Tensor, task: str) -> torch.Tensor:
    if task == "classification":
        return targets.long().view(-1)
    return targets.float()

# build the optimizer for the model
def build_optimizer(model: nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    params = model.parameters()
    if config.optimizer == "sgd":
        return torch.optim.SGD(params, lr=config.learning_rate)
    if config.optimizer == "adam":
        return torch.optim.Adam(params, lr=config.learning_rate)
    raise ValueError(f"Unsupported optimizer: {config.optimizer!r}")

# train the model for one epoch
def train_one_epoch(
    model: nn.Module, # the model to train
    loader: DataLoader,
    optimizer: torch.optim.Optimizer, # the optimizer to use for the training
    criterion: nn.Module, # the criterion to use for the training
    device: torch.device, # the device to use for the training
    *,
    task: str = "regression",
    step: int | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    n_samples = 0

    for features, targets in loader:
        features = features.to(device)
        targets = _prepare_targets(targets.to(device), task)
        optimizer.zero_grad()
        preds = model(features)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()
        batch_size = features.size(0)
        total_loss += loss.item() * batch_size
        n_samples += batch_size

    train_loss = total_loss / n_samples
    log_metrics({"loss": train_loss}, step=step, prefix="train_")
    return train_loss

# evaluate the model on the validation set
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, *, task: str = "regression", step: int | None = None, metric_prefix: str = "val_") -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    n_samples = 0
    all_targets: list[torch.Tensor] = []
    all_logits: list[torch.Tensor] = []

    for features, targets in loader:
        features = features.to(device)
        prepared_targets = _prepare_targets(targets.to(device), task)
        preds = model(features)
        loss = criterion(preds, prepared_targets)
        batch_size = features.size(0)
        total_loss += loss.item() * batch_size
        n_samples += batch_size

        if task == "classification":
            all_targets.append(prepared_targets.cpu())
            all_logits.append(preds.cpu())

    metrics: dict[str, float] = {"loss": total_loss / n_samples}

    if task == "classification":
        y_true = torch.cat(all_targets)
        logits = torch.cat(all_logits)
        metrics.update(compute_classification_metrics(y_true, logits))

    log_metrics(metrics, step=step, prefix=metric_prefix)
    return metrics

# fit the model to the data
def fit(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, optimizer: torch.optim.Optimizer, config: TrainConfig, *, criterion: nn.Module | None = None) -> dict[str, list[float]]:
    resolved = resolve_device(config.device)
    model.to(resolved)

    if criterion is None:
        criterion = resolve_criterion(config)

    log_train_config(config)

    # initialize the history of the training
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    for epoch in range(config.epochs):
        # train the model for one epoch
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, resolved, task=config.task, step=epoch)
        # evaluate the model on the validation set
        val_metrics = evaluate(model, val_loader, criterion, resolved, task=config.task, step=epoch, metric_prefix="val_")
        # update the history of the training
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])

        for key, value in val_metrics.items():
            if key == "loss":
                continue
            history_key = f"val_{key}"
            history.setdefault(history_key, []).append(value)

    if is_mlflow_active():
        log_learning_curve(history)
        log_pytorch_model(model)

    return history

# train the model
def train(config: TrainConfig) -> dict[str, list[float]]:
    config.validate()
    torch.manual_seed(config.seed)

    train_loader, val_loader, _ = create_surface_split_dataloaders(n=config.n_samples, batch_size=config.batch_size, seed=config.seed)
    model = MLP(config.layer_sizes, activation=config.activation, dropout=config.dropout)
    optimizer = build_optimizer(model, config)
    return fit(model, train_loader, val_loader, optimizer, config)