from __future__ import annotations
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def resolve_device(device: str | torch.device | None = None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)

# run one training epoch and return mean MSE over the loader
# calls optimizer.zero_grad() before each forward pass

# unlike the NumPy reference where backprop() returns fresh dW each step, PyTorch accumulates
# gradients in .grad across backward() calls until they are cleared
def train_one_epoch(
    model: nn.Module, # the model to train
    loader: DataLoader, # the data loader to use
    optimizer: torch.optim.Optimizer, # the optimizer to use
    criterion: nn.Module, # the loss function to use
    device: torch.device, # the device to use
) -> float: # the mean MSE over the loader
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

# run one evaluation epoch and return mean MSE over the loader
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

# train the model for a given number of epochs and return the history
# the history is a dictionary with the keys "train_loss" and "val_loss"
# the values are lists of floats containing the loss for each epoch
def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    *,
    device: str | torch.device | None = None,
    criterion: nn.Module | None = None,
) -> dict[str, list[float]]:
    resolved = resolve_device(device)
    model.to(resolved)
    if criterion is None:
        criterion = nn.MSELoss()

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    for _ in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, resolved)
        val_loss = evaluate(model, val_loader, criterion, resolved)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

    return history