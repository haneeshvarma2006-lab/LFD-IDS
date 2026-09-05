"""BOOTKCD and BAGKCD -- the resampling-aware KCD variants (Module 4).

Plain :class:`~lfd_ids.module4_defense.kcd.KMeansClusteringDefence` runs one
clustering pass over the whole poisoned training set.  The two variants below
mirror the structure of the attack they answer:

``BOOTKCD``
    Runs KCD on ``B`` bootstrap resamples and keeps, for each sample, the label
    that won a majority of the passes in which it appeared.  This matches the
    distribution-level noise BOOTLFA injects: a resample the adversary hit hard
    is outvoted by the ones it did not.
``BAGKCD``
    Runs KCD on ``B`` bagging subsets and votes the same way.  Because BAGLFA
    spreads corruption evenly, every subset carries roughly the same noise, and
    the vote suppresses the part of it that clustering can localise.

Both fall back to the single-pass repair for any sample no resample covered.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import DefenceConfig
from .kcd import DefenceResult, KMeansClusteringDefence


@dataclass
class _VoteTally:
    """Accumulates per-sample votes for the ``malicious`` label."""

    malicious: np.ndarray
    total: np.ndarray

    @classmethod
    def empty(cls, n: int) -> "_VoteTally":
        return cls(np.zeros(n, dtype=float), np.zeros(n, dtype=float))

    def add(self, indices: np.ndarray, labels: np.ndarray) -> None:
        np.add.at(self.malicious, indices, labels.astype(float))
        np.add.at(self.total, indices, 1.0)

    def resolve(self, fallback: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Majority label per sample; ``fallback`` covers uncovered samples."""
        covered = self.total > 0
        out = np.asarray(fallback, dtype=int).copy()
        share = np.divide(
            self.malicious, self.total, out=np.zeros_like(self.malicious), where=covered
        )
        out[covered] = (share[covered] >= 0.5).astype(int)
        return out, covered


class _ResamplingKCD(KMeansClusteringDefence):
    """Shared voting machinery for the two resampling-aware variants."""

    name = "ResamplingKCD"
    with_replacement = True

    def __init__(
        self,
        cfg: DefenceConfig,
        n_estimators: int = 10,
        subset_fraction: float = 0.8,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(cfg, rng)
        self.n_estimators = int(n_estimators)
        self.subset_fraction = float(subset_fraction)

    def _draw(self, n: int) -> np.ndarray:  # pragma: no cover - overridden
        raise NotImplementedError

    def defend(
        self,
        X: np.ndarray,
        y_poisoned: np.ndarray,
        trusted_index: np.ndarray | None = None,
        trusted_y: np.ndarray | None = None,
    ) -> DefenceResult:
        X = np.asarray(X, dtype=float)
        y_poisoned = np.asarray(y_poisoned, dtype=int)
        n = y_poisoned.size

        # The single-pass repair supplies the fallback labels and the geometry
        # (centroids, distances, thresholds) reported for the whole dataset.
        base = super().defend(X, y_poisoned, trusted_index, trusted_y)

        trusted_lookup = None
        if trusted_index is not None and trusted_y is not None and len(trusted_index) > 0:
            trusted_lookup = np.full(n, -1, dtype=int)
            trusted_lookup[np.asarray(trusted_index, dtype=int)] = np.asarray(trusted_y, dtype=int)

        tally = _VoteTally.empty(n)
        methods: list[str] = []
        for _ in range(self.n_estimators):
            draw = self._draw(n)
            sub_trusted_idx = sub_trusted_y = None
            if trusted_lookup is not None:
                local = np.flatnonzero(trusted_lookup[draw] >= 0)
                if local.size:
                    sub_trusted_idx = local
                    sub_trusted_y = trusted_lookup[draw][local]
            try:
                sub = super().defend(
                    X[draw], y_poisoned[draw], sub_trusted_idx, sub_trusted_y
                )
            except ValueError:
                # A degenerate resample (e.g. fewer rows than clusters) is skipped.
                continue
            methods.append(sub.assignment_method)
            tally.add(draw, sub.y_repaired)

        y_repaired, covered = tally.resolve(base.y_repaired)
        if trusted_lookup is not None:
            known = trusted_lookup >= 0
            y_repaired[known] = trusted_lookup[known]

        return DefenceResult(
            name=self.name,
            y_repaired=y_repaired,
            y_input=y_poisoned,
            cluster_labels=base.cluster_labels,
            centroids=base.centroids,
            distances=base.distances,
            thresholds=base.thresholds,
            cluster_to_class=base.cluster_to_class,
            relabel_mask=base.relabel_mask,
            assignment_method=base.assignment_method,
            keep_mask=base.keep_mask,
            sample_weight=base.sample_weight,
            metadata={
                **base.metadata,
                "n_estimators": self.n_estimators,
                "uncovered_fraction": float(np.mean(~covered)),
                "subset_assignment_methods": sorted(set(methods)),
                "single_pass_relabelled": base.n_relabelled,
            },
        )


class BOOTKCD(_ResamplingKCD):
    """KCD voted over bootstrap resamples -- the answer to BOOTLFA."""

    name = "BOOTKCD"

    def _draw(self, n: int) -> np.ndarray:
        return self.rng.integers(0, n, size=n)


class BAGKCD(_ResamplingKCD):
    """KCD voted over bagging subsets -- the answer to BAGLFA."""

    name = "BAGKCD"

    def _draw(self, n: int) -> np.ndarray:
        size = max(self.cfg.n_clusters, int(round(self.subset_fraction * n)))
        size = min(size, n)
        return self.rng.choice(n, size=size, replace=False)


#: Which defence variant answers which attack.
DEFENCE_FOR_ATTACK: dict[str, str] = {
    "BOOTLFA": "BOOTKCD",
    "BAGLFA": "BAGKCD",
}

DEFENCE_REGISTRY: dict[str, type[KMeansClusteringDefence]] = {
    "KCD": KMeansClusteringDefence,
    "BOOTKCD": BOOTKCD,
    "BAGKCD": BAGKCD,
}


def build_defence(
    name: str,
    cfg: DefenceConfig,
    rng: np.random.Generator,
    n_estimators: int = 10,
    subset_fraction: float = 0.8,
) -> KMeansClusteringDefence:
    """Instantiate ``KCD``, ``BOOTKCD`` or ``BAGKCD``."""
    key = name.upper()
    if key not in DEFENCE_REGISTRY:
        raise ValueError(f"unknown defence '{name}'; expected one of {sorted(DEFENCE_REGISTRY)}")
    cls = DEFENCE_REGISTRY[key]
    if cls is KMeansClusteringDefence:
        return cls(cfg, rng)
    return cls(cfg, n_estimators=n_estimators, subset_fraction=subset_fraction, rng=rng)  # type: ignore[call-arg]
