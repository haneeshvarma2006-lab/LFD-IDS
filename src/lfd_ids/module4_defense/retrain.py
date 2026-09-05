"""Model retraining, robustness validation and deployment (Module 4, block 3).

After KCD produces the cleansed dataset ``D''`` the IDS is retrained on it and
re-checked against the clean validation split.  A hardened model is published
to the fleet only if it clears the robustness gate; otherwise the incident is
recorded and the previously deployed model stays in service.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..module3_ids.evaluate import Metrics


@dataclass
class RobustnessVerdict:
    """The outcome of the robustness re-check on a retrained model."""

    passed: bool
    reasons: list[str] = field(default_factory=list)
    checks: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"passed": self.passed, "reasons": list(self.reasons), "checks": dict(self.checks)}


class RobustnessValidator:
    """Gate a retrained model on accuracy, FNR and AUC before deployment.

    The thresholds are expressed relative to the clean baseline, so the gate
    means "the hardened model is close enough to an unpoisoned one to ship",
    not "the hardened model beats an arbitrary constant".
    """

    def __init__(
        self,
        min_accuracy: float = 0.90,
        max_accuracy_drop: float = 0.05,
        max_fnr: float = 0.10,
        min_auc: float = 0.90,
    ) -> None:
        self.min_accuracy = min_accuracy
        self.max_accuracy_drop = max_accuracy_drop
        self.max_fnr = max_fnr
        self.min_auc = min_auc

    def validate(self, candidate: Metrics, baseline: Metrics | None = None) -> RobustnessVerdict:
        reasons: list[str] = []
        checks: dict = {
            "accuracy": candidate.accuracy,
            "fnr": candidate.fnr,
            "auc": candidate.auc,
            "min_accuracy": self.min_accuracy,
            "max_fnr": self.max_fnr,
            "min_auc": self.min_auc,
        }
        if candidate.accuracy < self.min_accuracy:
            reasons.append(
                f"accuracy {candidate.accuracy:.4f} below floor {self.min_accuracy:.2f}"
            )
        if candidate.fnr > self.max_fnr:
            reasons.append(f"false-negative rate {candidate.fnr:.4f} above cap {self.max_fnr:.2f}")
        if np.isfinite(candidate.auc) and candidate.auc < self.min_auc:
            reasons.append(f"AUC {candidate.auc:.4f} below floor {self.min_auc:.2f}")
        if baseline is not None:
            drop = baseline.accuracy - candidate.accuracy
            checks["baseline_accuracy"] = baseline.accuracy
            checks["accuracy_drop"] = drop
            checks["max_accuracy_drop"] = self.max_accuracy_drop
            if drop > self.max_accuracy_drop:
                reasons.append(
                    f"accuracy {drop:.4f} below the clean baseline "
                    f"(cap {self.max_accuracy_drop:.2f})"
                )
        return RobustnessVerdict(passed=not reasons, reasons=reasons, checks=checks)


@dataclass
class DeployedModel:
    """A model published to the connected-vehicle fleet."""

    version: str
    accuracy: float
    fnr: float
    auc: float
    trained_on: str

    def as_dict(self) -> dict:
        return {
            "version": self.version,
            "accuracy": self.accuracy,
            "fnr": self.fnr,
            "auc": self.auc,
            "trained_on": self.trained_on,
        }


class CloudModelPublisher:
    """Keeps the deployment history and the currently live model."""

    def __init__(self) -> None:
        self.history: list[DeployedModel] = []
        self.rejected: list[dict] = []

    @property
    def current(self) -> DeployedModel | None:
        return self.history[-1] if self.history else None

    def publish(
        self,
        version: str,
        metrics: Metrics,
        trained_on: str,
        verdict: RobustnessVerdict,
    ) -> DeployedModel | None:
        """Deploy the model if it passed the gate; otherwise log the rejection."""
        if not verdict.passed:
            self.rejected.append(
                {"version": version, "trained_on": trained_on, **verdict.as_dict()}
            )
            return None
        model = DeployedModel(
            version=version,
            accuracy=metrics.accuracy,
            fnr=metrics.fnr,
            auc=metrics.auc,
            trained_on=trained_on,
        )
        self.history.append(model)
        return model

    def summary(self) -> dict:
        return {
            "deployed": [m.as_dict() for m in self.history],
            "rejected": list(self.rejected),
            "current_version": self.current.version if self.current else None,
        }


def select_trusted_anchors(
    y: np.ndarray,
    fraction: float,
    rng: np.random.Generator,
    stratify: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Hold back a small clean subset for cluster-to-class assignment.

    Models the operator's ability to collect a limited number of records under
    attestation (a workshop-verified vehicle, a sealed test fleet) that the
    adversary cannot relabel.  Returns ``(indices, labels)``.
    """
    y = np.asarray(y, dtype=int)
    n = y.size
    k = int(round(fraction * n))
    if k <= 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    if k >= n:
        idx = np.arange(n)
        return idx, y.copy()
    if not stratify:
        idx = np.sort(rng.choice(n, size=k, replace=False))
        return idx, y[idx].copy()
    picks: list[np.ndarray] = []
    classes = np.unique(y)
    for cls in classes:
        pool = np.flatnonzero(y == cls)
        # At least one anchor per class, so neither cluster is left unlabelled.
        take = max(1, int(round(k * pool.size / n)))
        take = min(take, pool.size)
        picks.append(rng.choice(pool, size=take, replace=False))
    idx = np.sort(np.concatenate(picks))
    return idx, y[idx].copy()


def sequence_versions(prefix: str, items: Sequence[str]) -> list[str]:
    """Build deterministic model version strings such as ``ids-v1-baseline``."""
    return [f"{prefix}-v{i + 1}-{name}" for i, name in enumerate(items)]
