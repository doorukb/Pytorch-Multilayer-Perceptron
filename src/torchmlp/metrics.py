from __future__ import annotations
import numpy as np
import torch
from sklearn.metrics import accuracy_score as sklearn_accuracy_score
from sklearn.metrics import f1_score as sklearn_f1_score
from sklearn.metrics import roc_auc_score

# convert the tensor to a numpy array
def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

# compute the accuracy of the model
def accuracy(y_true: torch.Tensor | np.ndarray, y_pred: torch.Tensor | np.ndarray) -> float:
    return float(sklearn_accuracy_score(_to_numpy(y_true), _to_numpy(y_pred)))

# compute the f1 score of the model
def f1_score(y_true: torch.Tensor | np.ndarray, y_pred: torch.Tensor | np.ndarray, *, positive_class: int = 1) -> float:
    return float(sklearn_f1_score(_to_numpy(y_true), _to_numpy(y_pred), pos_label=positive_class, zero_division=0.0))

# compute the area under the curve of the model
def binary_auc(y_true: torch.Tensor | np.ndarray, y_scores: torch.Tensor | np.ndarray) -> float:
    y_true_np = _to_numpy(y_true)
    y_scores_np = _to_numpy(y_scores)
    try:
        score = float(roc_auc_score(y_true_np, y_scores_np))
    except ValueError:
        return 0.5
    if np.isnan(score):
        return 0.5
    return score

# compute the classification metrics of the model
def compute_classification_metrics(y_true: torch.Tensor, logits: torch.Tensor, *, positive_class: int = 1) -> dict[str, float]:
    probs = torch.softmax(logits, dim=-1)
    preds = torch.argmax(logits, dim=-1)
    scores = probs[:, positive_class]

    return {
        "accuracy": accuracy(y_true, preds),
        "f1": f1_score(y_true, preds, positive_class=positive_class),
        "auc": binary_auc(y_true, scores),
    }