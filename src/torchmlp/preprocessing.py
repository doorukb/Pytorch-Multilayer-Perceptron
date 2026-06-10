from __future__ import annotations
import torch

# per-column standardization fit on training features only
# regression targets (Z) are left unscaled for MSE
class FeatureScaler:
    def __init__(self) -> None:
        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None

    def fit(self, features: torch.Tensor) -> FeatureScaler:
        self.mean = features.mean(dim=0)
        self.std = features.std(dim=0).clamp_min(1e-8)
        return self

    def transform(self, features: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise RuntimeError("FeatureScaler must be fit before transform")
        return (features - self.mean) / self.std

    def fit_transform(self, features: torch.Tensor) -> torch.Tensor:
        return self.fit(features).transform(features)

# shuffle indices into disjoint train / val / test splits (80/10/10 on synthetic surface)
def random_split_indices(n: int, train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1, *, seed: int | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    total = train_ratio + val_ratio + test_ratio

    if not torch.isclose(torch.tensor(total), torch.tensor(1.0)):
        raise ValueError("train_ratio, val_ratio, and test_ratio must sum to 1.0")

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    perm = torch.randperm(n, generator=generator)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val : n_train + n_val + n_test]
    return train_idx, val_idx, test_idx