from __future__ import annotations
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmlp.config import TrainConfig
from torchmlp.data import create_surface_split_dataloaders
from torchmlp.model import MLP

# resolve the device to use for the training
def resolve_device(device: str | torch.device | None = None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)

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
    loader: DataLoader, # the data loader to use for the training
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module, # the criterion to use for the training
    device: torch.device, # the device to use for the training
) -> float:
    model.train()
    total_loss = 0.0
    n_samples = 0

    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        preds = model(features)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()
        batch_size = features.size(0)
        total_loss += loss.item() * batch_size
        n_samples += batch_size

    return total_loss / n_samples

# evaluate the model on the validation set
@torch.no_grad()

def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    n_samples = 0

    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)
        preds = model(features)
        loss = criterion(preds, targets)
        batch_size = features.size(0)
        total_loss += loss.item() * batch_size
        n_samples += batch_size

    return total_loss / n_samples

# fit the model to the data
def fit(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, optimizer: torch.optim.Optimizer, config: TrainConfig, *, criterion: nn.Module | None = None) -> dict[str, list[float]]:
    resolved = resolve_device(config.device)
    model.to(resolved)

    if criterion is None:
        criterion = nn.MSELoss()

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    for _ in range(config.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, resolved)
        val_loss = evaluate(model, val_loader, criterion, resolved)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

    return history

def train(config: TrainConfig) -> dict[str, list[float]]:
    config.validate()
    torch.manual_seed(config.seed)

    train_loader, val_loader, _ = create_surface_split_dataloaders(n=config.n_samples, batch_size=config.batch_size, seed=config.seed)

    model = MLP(config.layer_sizes, activation=config.activation, dropout=config.dropout)

    optimizer = build_optimizer(model, config)

    return fit(model, train_loader, val_loader, optimizer, config)