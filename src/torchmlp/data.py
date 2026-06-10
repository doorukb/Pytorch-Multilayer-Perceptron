from __future__ import annotations
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchmlp.preprocessing import FeatureScaler, random_split_indices

# sample points from the synthetic surface Z = X² - Y² + 1.2 + noise
# return features (n, 2) and targets (n, 1) as float32 tensors
def sample_surface_points(n: int, *, generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:

    x = torch.rand(n, generator=generator) * 2.0 - 1.0
    y = torch.rand(n, generator=generator) * 2.0 - 1.0
    noise = torch.randn(n, generator=generator) * 0.5
    z = x**2 - y**2 + 1.2 + noise
    features = torch.stack([x, y], dim=1)
    targets = z.unsqueeze(1)
    return features.float(), targets.float()

# synthetic regression dataset for Z = X² - Y² + 1.2 + noise
# Numpy pipelines store (n, 3) arrays and slice inputs with data[:, :2]
# this dataset returns explicit (features, target) tuples per index
class SurfaceDataset(Dataset):
    def __init__(self, n: int, *, generator: torch.Generator | None = None) -> None:
        self.features, self.targets = sample_surface_points(n, generator=generator)

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]

# make a reproducible generator for sampling random numbers
def _make_generator(seed: int | None) -> torch.Generator | None:
    if seed is None:
        return None
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen

# create synthetic regression datasets for training and testing
def create_surface_datasets(
    train_size: int = 100, # 100 training points so that training is fast
    test_size: int = 20, # 20 test points so that validation is reliable
    *, seed: int | None = None) -> tuple[SurfaceDataset, SurfaceDataset]:
    train_gen = _make_generator(seed)
    test_gen = _make_generator(None if seed is None else seed + 1)
    train = SurfaceDataset(train_size, generator=train_gen)
    test = SurfaceDataset(test_size, generator=test_gen)
    return train, test

# the data loader wraps the dataset and provides an iterator over batches
def create_surface_dataloaders(train_size: int = 100, test_size: int = 20, batch_size: int = 32, *, seed: int | None = None) -> tuple[DataLoader, DataLoader]:
    train_ds, test_ds = create_surface_datasets(train_size, test_size, seed=seed)
    loader_gen = _make_generator(None if seed is None else seed + 2)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=loader_gen)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# sample surface points and split them into train, validation and test sets
def create_surface_split_datasets(n: int = 1000, *, seed: int = 42) -> tuple[TensorDataset, TensorDataset, TensorDataset]:
    sample_gen = _make_generator(seed)
    features, targets = sample_surface_points(n, generator=sample_gen)
    train_idx, val_idx, test_idx = random_split_indices(n, seed=seed)

    scaler = FeatureScaler()
    x_train = scaler.fit_transform(features[train_idx])
    x_val = scaler.transform(features[val_idx])
    x_test = scaler.transform(features[test_idx])

    return (
        TensorDataset(x_train, targets[train_idx]),
        TensorDataset(x_val, targets[val_idx]),
        TensorDataset(x_test, targets[test_idx]),
    )

def create_surface_split_dataloaders(n: int = 1000, batch_size: int = 32, *, seed: int = 42) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_ds, val_ds, test_ds = create_surface_split_datasets(n, seed=seed)
    loader_gen = _make_generator(None if seed is None else seed + 2)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=loader_gen)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader