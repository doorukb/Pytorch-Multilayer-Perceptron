import pytest
import torch
from torchmlp.preprocessing import FeatureScaler, random_split_indices

def test_random_split_sizes():
    train_idx, val_idx, test_idx = random_split_indices(1000, seed=42)
    assert len(train_idx) == 800
    assert len(val_idx) == 100
    assert len(test_idx) == 100

    all_idx = torch.cat([train_idx, val_idx, test_idx])
    assert len(torch.unique(all_idx)) == 1000

def test_random_split_reproducible():
    split_a = random_split_indices(100, seed=7)
    split_b = random_split_indices(100, seed=7)
    for a, b in zip(split_a, split_b, strict=True):
        assert torch.equal(a, b)

def test_feature_scaler_train_zero_mean_unit_std():
    features = torch.randn(200, 2) * 5.0 + 3.0
    scaler = FeatureScaler()
    scaled = scaler.fit_transform(features)
    assert torch.allclose(scaled.mean(dim=0), torch.zeros(2), atol=1e-5)
    assert torch.allclose(scaled.std(dim=0), torch.ones(2), atol=1e-5)

def test_feature_scaler_no_leakage():
    features = torch.randn(300, 2) * 2.0 + 1.0
    train_idx, val_idx, _ = random_split_indices(300, seed=0)

    scaler = FeatureScaler()
    scaler.fit(features[train_idx])
    val_scaled = scaler.transform(features[val_idx])
    assert not torch.allclose(val_scaled.mean(dim=0), torch.zeros(2), atol=0.05)

def test_invalid_split_ratios_raises():
    with pytest.raises(ValueError, match="must sum to 1.0"):
        random_split_indices(100, train_ratio=0.5, val_ratio=0.3, test_ratio=0.1)