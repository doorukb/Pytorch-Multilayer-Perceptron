from __future__ import annotations
import numpy as np
import torch

# compute the accuracy of the model
def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))

# compute the f1 score of the model
def f1_score(y_true: np.ndarray, y_pred: np.ndarray, *, positive_class: int = 1) -> float:
    tp = int(np.sum((y_true == positive_class) & (y_pred == positive_class)))
    fp = int(np.sum((y_true != positive_class) & (y_pred == positive_class)))
    fn = int(np.sum((y_true == positive_class) & (y_pred != positive_class)))
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0.0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))

# compute the area under the curve of the model
def binary_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    classes = np.unique(y_true)
    if classes.size < 2:
        return 0.5

    order = np.argsort(y_scores)
    y_sorted = y_true[order]
    n_pos = int(np.sum(y_sorted == 1))
    n_neg = int(np.sum(y_sorted == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.5

    ranks = np.arange(1, len(y_sorted) + 1)
    sum_ranks_pos = float(np.sum(ranks[y_sorted == 1]))
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))

# compute the classification metrics of the model
def compute_classification_metrics(y_true: torch.Tensor, logits: torch.Tensor, *, positive_class: int = 1) -> dict[str, float]:
    probs = torch.softmax(logits, dim=-1)
    preds = torch.argmax(logits, dim=-1)

    y_np = y_true.detach().cpu().numpy().astype(np.int64)
    pred_np = preds.detach().cpu().numpy().astype(np.int64)
    scores_np = probs[:, positive_class].detach().cpu().numpy()

    return {
        "accuracy": accuracy(y_np, pred_np),
        "f1": f1_score(y_np, pred_np, positive_class=positive_class),
        "auc": binary_auc(y_np, scores_np),
    }