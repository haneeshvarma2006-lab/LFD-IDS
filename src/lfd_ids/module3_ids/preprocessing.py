"""Preprocessing (Module 3, block 1).

Splits the labelled dataset 60/20/20 into train/validation/test and normalises
the feature vector.  The scaler is fitted on the *training* split only, so no
information leaks from validation or test into the model.

The split is performed once on the clean dataset and reused for every attack
and defence run, which is what makes the accuracy numbers across poisoning
intensities directly comparable: only the training *labels* change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..config import PreprocessConfig
from ..module1_acquisition.database import LabelledDataset


@dataclass
class Scaler:
    """Min-max or zero-mean/unit-variance normaliser fitted on the train split."""

    kind: str
    offset: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray, kind: str = "minmax") -> "Scaler":
        X = np.asarray(X, dtype=float)
        if kind == "minmax":
            offset = X.min(axis=0)
            span = X.max(axis=0) - offset
            scale = np.where(span > 1e-12, span, 1.0)
        elif kind == "standard":
            offset = X.mean(axis=0)
            std = X.std(axis=0)
            scale = np.where(std > 1e-12, std, 1.0)
        else:
            raise ValueError(f"unknown scaler '{kind}'; expected minmax|standard")
        return cls(kind=kind, offset=offset, scale=scale)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=float) - self.offset) / self.scale

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float) * self.scale + self.offset


@dataclass
class DataSplit:
    """Normalised train/validation/test tensors plus their provenance."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    scaler: Scaler
    feature_names: tuple[str, ...]
    train_index: np.ndarray
    val_index: np.ndarray
    test_index: np.ndarray
    attack_types_test: np.ndarray | None = None

    def summary(self) -> dict:
        def counts(y: np.ndarray) -> dict[str, int]:
            values, n = np.unique(y, return_counts=True)
            return {str(int(v)): int(c) for v, c in zip(values, n)}

        return {
            "n_train": int(self.y_train.size),
            "n_val": int(self.y_val.size),
            "n_test": int(self.y_test.size),
            "n_features": int(self.X_train.shape[1]),
            "feature_names": list(self.feature_names),
            "scaler": self.scaler.kind,
            "class_counts": {
                "train": counts(self.y_train),
                "val": counts(self.y_val),
                "test": counts(self.y_test),
            },
        }

    def with_train_labels(self, y_train: np.ndarray) -> "DataSplit":
        """Return a copy whose training labels are replaced (poisoned/cleansed)."""
        y_train = np.asarray(y_train, dtype=int)
        if y_train.size != self.y_train.size:
            raise ValueError(
                f"expected {self.y_train.size} training labels, got {y_train.size}"
            )
        return DataSplit(
            X_train=self.X_train,
            y_train=y_train,
            X_val=self.X_val,
            y_val=self.y_val,
            X_test=self.X_test,
            y_test=self.y_test,
            scaler=self.scaler,
            feature_names=self.feature_names,
            train_index=self.train_index,
            val_index=self.val_index,
            test_index=self.test_index,
            attack_types_test=self.attack_types_test,
        )


def stratified_split(
    y: np.ndarray,
    fractions: Sequence[float],
    rng: np.random.Generator,
    stratify: bool = True,
) -> list[np.ndarray]:
    """Partition indices into groups sized by ``fractions``.

    With ``stratify`` the class balance of ``y`` is preserved in every group.
    """
    y = np.asarray(y)
    n = y.size
    groups: list[list[np.ndarray]] = [[] for _ in fractions]
    strata = [np.flatnonzero(y == c) for c in np.unique(y)] if stratify else [np.arange(n)]
    for stratum in strata:
        idx = rng.permutation(stratum)
        # Cut points from the cumulative fractions, so the pieces always
        # partition the stratum exactly.
        bounds = np.round(np.cumsum(fractions) * idx.size).astype(int)
        bounds[-1] = idx.size
        start = 0
        for g, end in enumerate(bounds):
            groups[g].append(idx[start:end])
            start = end
    return [np.sort(np.concatenate(parts)) for parts in groups]


def preprocess(
    dataset: LabelledDataset,
    cfg: PreprocessConfig,
    rng: np.random.Generator,
) -> DataSplit:
    """Split 60/20/20 and normalise, fitting the scaler on the train split."""
    fractions = (cfg.train_frac, cfg.val_frac, cfg.test_frac)
    train_idx, val_idx, test_idx = stratified_split(dataset.y, fractions, rng, cfg.stratify)
    scaler = Scaler.fit(dataset.X[train_idx], cfg.scaler)
    attack_types_test = (
        dataset.attack_types[test_idx] if dataset.attack_types is not None else None
    )
    return DataSplit(
        X_train=scaler.transform(dataset.X[train_idx]),
        y_train=dataset.y[train_idx].copy(),
        X_val=scaler.transform(dataset.X[val_idx]),
        y_val=dataset.y[val_idx].copy(),
        X_test=scaler.transform(dataset.X[test_idx]),
        y_test=dataset.y[test_idx].copy(),
        scaler=scaler,
        feature_names=dataset.feature_names,
        train_index=train_idx,
        val_index=val_idx,
        test_index=test_idx,
        attack_types_test=attack_types_test,
    )
