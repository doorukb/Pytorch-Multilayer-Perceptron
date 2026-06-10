import torch
from torchmlp.data import (
    SurfaceDataset,
    create_surface_dataloaders,
    create_surface_datasets,
    create_surface_split_dataloaders,
    create_surface_split_datasets,
    sample_surface_points,
)

def test_sample_surface_points_shape():
    features, targets = sample_surface_points(50)
    assert features.shape == (50, 2)
    assert targets.shape == (50, 1)
    assert features.dtype == torch.float32
    assert targets.dtype == torch.float32

def test_sample_surface_points_z_relationship():
    features, targets = sample_surface_points(5000)
    x, y = features[:, 0], features[:, 1]
    z = targets.squeeze(1)
    residual = z - (x**2 - y**2 + 1.2)
    assert abs(residual.mean().item()) < 0.1
    assert 0.3 < residual.std().item() < 0.7

def test_surface_dataset_len_and_item():
    ds = SurfaceDataset(25)
    assert len(ds) == 25
    features, target = ds[0]
    assert features.shape == (2,)
    assert target.shape == (1,)

def test_create_surface_datasets_sizes():
    train, test = create_surface_datasets(100, 20)
    assert len(train) == 100
    assert len(test) == 20
    assert train.features.shape == (100, 2)
    assert test.targets.shape == (20, 1)

def test_dataloader_batch_shapes():
    train_loader, _ = create_surface_dataloaders(train_size=100, test_size=20, batch_size=32)
    features, targets = next(iter(train_loader))
    assert features.shape[1] == 2
    assert targets.shape[1] == 1
    assert features.shape[0] <= 32

def test_dataloader_reproducible_with_seed():
    loader_a, _ = create_surface_dataloaders(train_size=64, batch_size=16, seed=42)
    loader_b, _ = create_surface_dataloaders(train_size=64, batch_size=16, seed=42)
    loader_c, _ = create_surface_dataloaders(train_size=64, batch_size=16, seed=99)

    x_a, y_a = next(iter(loader_a))
    x_b, y_b = next(iter(loader_b))
    x_c, y_c = next(iter(loader_c))

    assert torch.equal(x_a, x_b)
    assert torch.equal(y_a, y_b)
    assert not torch.equal(x_a, x_c) or not torch.equal(y_a, y_c)

def test_create_surface_split_datasets_sizes():
    train, val, test = create_surface_split_datasets(n=1000, seed=42)
    assert len(train) == 800
    assert len(val) == 100
    assert len(test) == 100
    features, target = train[0]
    assert features.shape == (2,)
    assert target.shape == (1,)

def test_create_surface_split_dataloaders_batch():
    train_loader, _, _ = create_surface_split_dataloaders(n=1000, batch_size=32, seed=42)
    features, targets = next(iter(train_loader))
    assert features.shape[1] == 2
    assert targets.shape[1] == 1
    assert features.shape[0] <= 32

def test_split_dataloaders_reproducible():
    loader_a, _, _ = create_surface_split_dataloaders(n=1000, batch_size=32, seed=42)
    loader_b, _, _ = create_surface_split_dataloaders(n=1000, batch_size=32, seed=42)
    loader_c, _, _ = create_surface_split_dataloaders(n=1000, batch_size=32, seed=99)

    x_a, y_a = next(iter(loader_a))
    x_b, y_b = next(iter(loader_b))
    x_c, y_c = next(iter(loader_c))

    assert torch.equal(x_a, x_b)
    assert torch.equal(y_a, y_b)
    assert not torch.equal(x_a, x_c) or not torch.equal(y_a, y_c)