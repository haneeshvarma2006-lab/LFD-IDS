"""Evaluation engine (Module 3, block 3).

Computes the confusion matrix and the metric set the project reports:
accuracy, precision, recall, F1, false-negative rate and ROC AUC.

The false-negative rate is the headline safety metric here -- a false negative
is an intrusion the IDS waved through -- which is why it is reported alongside
accuracy for every attack and defence configuration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np


@dataclass
class ConfusionMatrix:
    """Binary confusion matrix with ``malicious`` as the positive class."""

    tn: int
    fp: int
    fn: int
    tp: int

    @classmethod
    def from_predictions(cls, y_true: np.ndarray, y_pred: np.ndarray) -> "ConfusionMatrix":
        y_true = np.asarray(y_true, dtype=int)
        y_pred = np.asarray(y_pred, dtype=int)
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
            )
        return cls(
            tn=int(np.sum((y_true == 0) & (y_pred == 0))),
            fp=int(np.sum((y_true == 0) & (y_pred == 1))),
            fn=int(np.sum((y_true == 1) & (y_pred == 0))),
            tp=int(np.sum((y_true == 1) & (y_pred == 1))),
        )

    @property
    def total(self) -> int:
        return self.tn + self.fp + self.fn + self.tp

    def as_array(self) -> np.ndarray:
        """``[[TN, FP], [FN, TP]]`` -- rows are truth, columns are prediction."""
        return np.array([[self.tn, self.fp], [self.fn, self.tp]], dtype=int)

    def as_dict(self) -> dict:
        return asdict(self)


def _safe_div(numerator: float, denominator: float) -> float:
    """Ratio that returns 0.0 rather than NaN when the denominator vanishes."""
    return float(numerator / denominator) if denominator else 0.0


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """ROC AUC via the Mann-Whitney U statistic, with mid-ranks for ties.

    Equivalent to ``sklearn.metrics.roc_auc_score`` but keeps the metric layer
    free of a scikit-learn dependency.  Returns ``nan`` for a single-class
    ground truth, where AUC is undefined.
    """
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=float)
    i = 0
    while i < sorted_scores.size:
        j = i
        while j + 1 < sorted_scores.size and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        # Average rank (1-based) shared by this run of tied scores.
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    rank_sum_pos = float(np.sum(ranks[y_true == 1]))
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def roc_curve(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(fpr, tpr)`` sampled at every distinct score threshold."""
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind="mergesort")
    y_sorted = y_true[order]
    tps = np.cumsum(y_sorted == 1)
    fps = np.cumsum(y_sorted == 0)
    n_pos, n_neg = max(int(tps[-1]), 1), max(int(fps[-1]), 1)
    tpr = np.concatenate([[0.0], tps / n_pos])
    fpr = np.concatenate([[0.0], fps / n_neg])
    return fpr, tpr


@dataclass
class Metrics:
    """The metric set reported for every model in the experiment grid."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    fnr: float
    fpr: float
    specificity: float
    auc: float
    confusion: ConfusionMatrix
    n_samples: int

    def as_dict(self) -> dict:
        out = {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "fnr": self.fnr,
            "fpr": self.fpr,
            "specificity": self.specificity,
            "auc": self.auc,
            "n_samples": self.n_samples,
            "confusion": self.confusion.as_dict(),
        }
        return out

    def headline(self) -> str:
        return (
            f"acc={self.accuracy:.4f} prec={self.precision:.4f} rec={self.recall:.4f} "
            f"f1={self.f1:.4f} fnr={self.fnr:.4f} auc={self.auc:.4f}"
        )


def evaluate(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.5,
) -> Metrics:
    """Score probabilistic predictions against the clean ground truth."""
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    y_pred = (scores >= threshold).astype(int)
    cm = ConfusionMatrix.from_predictions(y_true, y_pred)
    precision = _safe_div(cm.tp, cm.tp + cm.fp)
    recall = _safe_div(cm.tp, cm.tp + cm.fn)
    return Metrics(
        accuracy=_safe_div(cm.tp + cm.tn, cm.total),
        precision=precision,
        recall=recall,
        f1=_safe_div(2 * precision * recall, precision + recall),
        fnr=_safe_div(cm.fn, cm.fn + cm.tp),
        fpr=_safe_div(cm.fp, cm.fp + cm.tn),
        specificity=_safe_div(cm.tn, cm.tn + cm.fp),
        auc=roc_auc(y_true, scores),
        confusion=cm,
        n_samples=int(y_true.size),
    )


def per_attack_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    attack_types: Sequence[str] | np.ndarray,
) -> dict[str, dict[str, float | int]]:
    """Detection rate broken down by the attack type that produced a sample."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    attack_types = np.asarray(attack_types, dtype=object)
    out: dict[str, dict[str, float | int]] = {}
    for name in np.unique(attack_types):
        if name == "none":
            continue
        mask = (attack_types == name) & (y_true == 1)
        if not mask.any():
            continue
        out[str(name)] = {
            "n": int(mask.sum()),
            "detected": int(np.sum(y_pred[mask] == 1)),
            "recall": float(np.mean(y_pred[mask] == 1)),
        }
    return out


def label_recovery_metrics(
    y_true: np.ndarray,
    y_poisoned: np.ndarray,
    y_repaired: np.ndarray,
) -> dict:
    """How well a defence undid the adversary's label inversions.

    ``corrected`` counts flipped labels the defence restored; ``newly_corrupted``
    counts clean labels the defence itself broke.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_poisoned = np.asarray(y_poisoned, dtype=int)
    y_repaired = np.asarray(y_repaired, dtype=int)
    was_flipped = y_poisoned != y_true
    still_wrong = y_repaired != y_true
    corrected = int(np.sum(was_flipped & ~still_wrong))
    missed = int(np.sum(was_flipped & still_wrong))
    newly_corrupted = int(np.sum(~was_flipped & still_wrong))
    n_flipped = int(was_flipped.sum())
    return {
        "n_train": int(y_true.size),
        "n_flipped_by_attack": n_flipped,
        "corrected": corrected,
        "missed": missed,
        "newly_corrupted": newly_corrupted,
        "correction_rate": _safe_div(corrected, n_flipped),
        "label_error_before": float(np.mean(was_flipped)),
        "label_error_after": float(np.mean(still_wrong)),
        "labels_changed": int(np.sum(y_repaired != y_poisoned)),
    }
