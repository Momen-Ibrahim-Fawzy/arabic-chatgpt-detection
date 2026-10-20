"""Binary classification metrics in plain NumPy (macro-averaged, as reported in the project)."""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np


def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int]) -> np.ndarray:
    """2x2 matrix, rows = true class (0 human, 1 ChatGPT), columns = predicted class."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    cm = np.zeros((2, 2), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def per_class(cm: np.ndarray) -> Dict[str, np.ndarray]:
    tp = np.diag(cm).astype(float)
    precision = np.array([_safe_div(tp[c], cm[:, c].sum()) for c in range(2)])
    recall = np.array([_safe_div(tp[c], cm[c, :].sum()) for c in range(2)])
    f1 = np.array([_safe_div(2 * p * r, p + r) for p, r in zip(precision, recall)])
    return {"precision": precision, "recall": recall, "f1": f1}


def roc_auc(y_true: Sequence[int], scores: Sequence[float]) -> float:
    """Area under the ROC curve via the Mann-Whitney U statistic (ties get half credit)."""
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    pos, neg = scores[y_true == 1], scores[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), dtype=float)
    values = np.concatenate([pos, neg])[order]
    i = 0
    while i < len(values):  # average ranks over ties
        j = i
        while j + 1 < len(values) and values[j + 1] == values[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    u = ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(pos) * len(neg)))


def classification_report(y_true: Sequence[int], y_pred: Sequence[int],
                          scores: Optional[Sequence[float]] = None) -> Dict[str, float]:
    """Accuracy, macro precision/recall/F1 and ROC-AUC (on ``scores`` if given, else hard labels)."""
    cm = confusion_matrix(y_true, y_pred)
    pc = per_class(cm)
    present = [c for c in range(2) if cm[c, :].sum() > 0]  # single-class sets: average present classes
    report = {
        "accuracy": _safe_div(np.trace(cm), cm.sum()),
        "precision": float(pc["precision"][present].mean()),
        "recall": float(pc["recall"][present].mean()),
        "f1_macro": float(pc["f1"][present].mean()),
        "roc_auc": roc_auc(y_true, scores if scores is not None else y_pred),
    }
    return report
