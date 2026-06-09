import torch
import torch.nn as nn
from torchmlp.data import create_surface_dataloaders
from torchmlp.model import MLP
from torchmlp.trainer import evaluate, fit, resolve_device, train_one_epoch

def _make_training_setup(seed: int = 42):
    torch.manual_seed(seed)
    model = MLP([2, 5, 1])
    train_loader, val_loader = create_surface_dataloaders(train_size=100, test_size=20, batch_size=32, seed=seed)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    device = torch.device("cpu")
    return model, train_loader, val_loader, optimizer, criterion, device

def test_fit_reduces_train_loss():
    model, train_loader, val_loader, optimizer, criterion, device = _make_training_setup()
    history = fit(model, train_loader, val_loader, optimizer, epochs=20, device=device, criterion=criterion)
    assert len(history["train_loss"]) == 20
    assert history["train_loss"][-1] < history["train_loss"][0]

def test_eval_runs_in_eval_mode():
    model, train_loader, val_loader, optimizer, criterion, device = _make_training_setup()
    model.to(device)
    evaluate(model, val_loader, criterion, device)
    assert model.training is False

def test_train_runs_in_train_mode():
    model, train_loader, _, optimizer, criterion, device = _make_training_setup()
    model.to(device)
    model.eval()
    train_one_epoch(model, train_loader, optimizer, criterion, device)
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
        model, train_loader, val_loader, optimizer, criterion, device = _make_training_setup(seed)
        return fit(model, train_loader, val_loader, optimizer, epochs=10, device=device, criterion=criterion)

    history_a = run_fit(42)
    history_b = run_fit(42)
    assert history_a["train_loss"] == history_b["train_loss"]
    assert history_a["val_loss"] == history_b["val_loss"]
