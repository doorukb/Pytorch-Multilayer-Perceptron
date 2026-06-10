import numpy as np
import torch
from torchmlp.metrics import (
    accuracy,
    binary_auc,
    compute_classification_metrics,
    f1_score,
)

def test_accuracy_perfect_predictions():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 0])
    assert accuracy(y_true, y_pred) == 1.0

def test_f1_perfect_positive_class():
    y_true = np.array([1, 1, 0, 1])
    y_pred = np.array([1, 1, 0, 1])
    assert f1_score(y_true, y_pred) == 1.0

def test_binary_auc_perfect_ranking():
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.8, 0.9])
    assert binary_auc(y_true, y_scores) == 1.0

def test_binary_auc_single_class_returns_half():
    assert binary_auc(np.array([1, 1, 1]), np.array([0.2, 0.5, 0.9])) == 0.5

def test_compute_classification_metrics_keys():
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0], [0.0, 2.0]])
    y_true = torch.tensor([0, 1, 0, 1])
    metrics = compute_classification_metrics(y_true, logits)
    assert set(metrics) == {"accuracy", "f1", "auc"}
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0

def test_metrics_accept_torch_tensors():
    y_true = torch.tensor([0, 1, 0, 1])
    y_pred = torch.tensor([0, 1, 0, 1])
    y_scores = torch.tensor([0.1, 0.9, 0.2, 0.8])

    acc = accuracy(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = binary_auc(y_true, y_scores)

    assert isinstance(acc, float)
    assert isinstance(f1, float)
    assert isinstance(auc, float)
    assert acc == 1.0
    assert f1 == 1.0
    assert auc == 1.0