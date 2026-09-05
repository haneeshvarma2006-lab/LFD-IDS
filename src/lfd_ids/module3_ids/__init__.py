"""Module 3 -- Deep-learning-based IDS classification."""

from .alert import Alert, AlertModule, BroadcastRecord
from .evaluate import (
    ConfusionMatrix,
    Metrics,
    evaluate,
    label_recovery_metrics,
    per_attack_recall,
    roc_auc,
    roc_curve,
)
from .model import (
    Classifier,
    KerasMLP,
    NumpyMLP,
    build_model,
    describe_architecture,
    keras_available,
    resolve_backend,
)
from .preprocessing import DataSplit, Scaler, preprocess, stratified_split

__all__ = [
    "Alert",
    "AlertModule",
    "BroadcastRecord",
    "Classifier",
    "ConfusionMatrix",
    "DataSplit",
    "KerasMLP",
    "Metrics",
    "NumpyMLP",
    "Scaler",
    "build_model",
    "describe_architecture",
    "evaluate",
    "keras_available",
    "label_recovery_metrics",
    "per_attack_recall",
    "preprocess",
    "resolve_backend",
    "roc_auc",
    "roc_curve",
    "stratified_split",
]
