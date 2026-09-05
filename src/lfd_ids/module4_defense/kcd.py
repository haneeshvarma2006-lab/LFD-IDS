"""K-means Clustering Defence -- KCD (Module 4, blocks 1 and 2).

The defence assumes the adversary corrupted *labels* but not *features*: the
feature vectors in the poisoned dataset D' are still the ones the sensors
reported.  So the geometry of D' still carries the class structure, and the
labels can be re-derived from it.

Procedure, as specified by the project:

1. Cluster the poisoned training features with k-means (``k = 2``).
2. Assign each cluster a class label (Cluster Label Assignment).
3. Measure the Euclidean distance ``d(x_i)`` of every sample from its assigned
   cluster centroid.
4. Estimate the mean distance ``mu``.
5. Relabel every sample with ``d(x_i) < mu`` to its cluster's label, leaving
   samples in the sparse outer shell untouched.

Step 5 is deliberately conservative: samples close to a centroid are the ones
whose cluster membership is most trustworthy, so those are the only labels the
defence is willing to overwrite.

Cluster-to-class assignment
---------------------------
Mapping a cluster onto ``benign``/``malicious`` needs a source of truth that
the adversary has not touched.  Three options are supported:

``majority``
    Vote of the (poisoned) labels inside each cluster.  This is the published
    rule and works whenever fewer than half the labels in a cluster were
    inverted -- but at ``p = 50%`` the vote is a coin flip.
``anchor``
    Vote of a small trusted subset the defender holds back from the poisoned
    pool (e.g. records collected under attestation).  Robust at any ``p``.
``auto`` (default)
    Use ``majority`` when its margin is decisive, otherwise fall back to
    ``anchor``.  This keeps the published behaviour where it is sound and
    degrades gracefully at high poisoning intensity.

A ``majority``-only run at ``p = 50%`` is a legitimate experiment; it just
measures a different, weaker defence, and the reported ``assignment_method``
records which rule actually fired.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import DefenceConfig
from .kmeans import KMeans, euclidean_distances


@dataclass
class DefenceResult:
    """The cleansed dataset ``D''`` plus the internals used to produce it."""

    name: str
    y_repaired: np.ndarray
    y_input: np.ndarray
    cluster_labels: np.ndarray
    centroids: np.ndarray
    distances: np.ndarray
    thresholds: np.ndarray
    cluster_to_class: dict[int, int]
    relabel_mask: np.ndarray
    assignment_method: str
    #: Rows of D'' the retrainer should use (all ``True`` unless samples were dropped).
    keep_mask: np.ndarray | None = None
    #: Per-sample training weights (``None`` means uniform).
    sample_weight: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def n_relabelled(self) -> int:
        return int(np.sum(self.y_repaired != self.y_input))

    def training_set(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Return ``(X'', y'', sample_weight)`` for retraining on D''."""
        X = np.asarray(X, dtype=float)
        if self.keep_mask is None:
            return X, self.y_repaired, self.sample_weight
        keep = np.asarray(self.keep_mask, dtype=bool)
        weight = None if self.sample_weight is None else self.sample_weight[keep]
        return X[keep], self.y_repaired[keep], weight

    def summary(self) -> dict:
        kept = int(self.keep_mask.sum()) if self.keep_mask is not None else int(self.y_input.size)
        return {
            "defence": self.name,
            "assignment_method": self.assignment_method,
            "cluster_to_class": {str(k): int(v) for k, v in self.cluster_to_class.items()},
            "n_within_threshold": int(self.relabel_mask.sum()),
            "n_relabelled": self.n_relabelled,
            "n_kept_for_retraining": kept,
            "mean_distance": float(self.distances.mean()),
            "thresholds": [float(t) for t in self.thresholds],
            **self.metadata,
        }


def assign_cluster_classes(
    cluster_labels: np.ndarray,
    y_poisoned: np.ndarray,
    n_clusters: int,
    method: str,
    majority_margin: float,
    trusted_index: np.ndarray | None = None,
    trusted_y: np.ndarray | None = None,
) -> tuple[dict[int, int], str, dict]:
    """Map each cluster id onto a class label.

    Returns the mapping, the rule that actually decided it, and diagnostics.
    """
    cluster_labels = np.asarray(cluster_labels, dtype=int)
    y_poisoned = np.asarray(y_poisoned, dtype=int)

    def vote(labels: np.ndarray, values: np.ndarray) -> tuple[dict[int, int], np.ndarray]:
        mapping: dict[int, int] = {}
        margins = np.zeros(n_clusters, dtype=float)
        for k in range(n_clusters):
            members = values[labels == k]
            if members.size == 0:
                mapping[k] = 0
                margins[k] = 0.0
                continue
            share = float(members.mean())
            mapping[k] = int(share >= 0.5)
            # 0.0 at a perfect 50/50 tie, 1.0 when unanimous.
            margins[k] = abs(2.0 * share - 1.0)
        return mapping, margins

    majority_map, margins = vote(cluster_labels, y_poisoned)
    degenerate = len(set(majority_map.values())) < min(n_clusters, 2)
    diagnostics = {
        "majority_margins": [float(m) for m in margins],
        "majority_degenerate": bool(degenerate),
    }

    have_anchor = trusted_index is not None and trusted_y is not None and len(trusted_index) > 0
    weak = bool(margins.min() < majority_margin) or degenerate

    if method == "majority" or (method == "auto" and not weak):
        return majority_map, "majority", diagnostics

    if not have_anchor:
        if method == "anchor":
            raise ValueError(
                "defence.label_assignment='anchor' requires a trusted subset; "
                "set defence.trusted_fraction > 0"
            )
        diagnostics["anchor_unavailable"] = True
        return majority_map, "majority_fallback", diagnostics

    anchor_map, anchor_margins = vote(
        cluster_labels[np.asarray(trusted_index, dtype=int)],
        np.asarray(trusted_y, dtype=int),
    )
    diagnostics["anchor_margins"] = [float(m) for m in anchor_margins]
    diagnostics["n_trusted"] = int(len(trusted_index))
    return anchor_map, "anchor", diagnostics


class KMeansClusteringDefence:
    """KCD: recover poisoned labels from the geometry of the feature space."""

    name = "KCD"

    def __init__(self, cfg: DefenceConfig, rng: np.random.Generator | None = None) -> None:
        self.cfg = cfg
        self.rng = rng or np.random.default_rng()

    def _thresholds(self, distances: np.ndarray, cluster_labels: np.ndarray) -> np.ndarray:
        """Mean-distance threshold ``mu``, per cluster or globally."""
        n_clusters = self.cfg.n_clusters
        if self.cfg.threshold_scope == "global":
            mu = float(distances.mean()) * self.cfg.threshold_scale
            return np.full(n_clusters, mu, dtype=float)
        if self.cfg.threshold_scope != "per_cluster":
            raise ValueError("defence.threshold_scope must be per_cluster|global")
        out = np.zeros(n_clusters, dtype=float)
        for k in range(n_clusters):
            members = distances[cluster_labels == k]
            out[k] = float(members.mean()) * self.cfg.threshold_scale if members.size else 0.0
        return out

    def defend(
        self,
        X: np.ndarray,
        y_poisoned: np.ndarray,
        trusted_index: np.ndarray | None = None,
        trusted_y: np.ndarray | None = None,
    ) -> DefenceResult:
        """Cleanse ``(X, y_poisoned)`` and return the repaired labels."""
        X = np.asarray(X, dtype=float)
        y_poisoned = np.asarray(y_poisoned, dtype=int)
        if X.shape[0] != y_poisoned.size:
            raise ValueError(
                f"X has {X.shape[0]} rows but y_poisoned has {y_poisoned.size} labels"
            )

        km = KMeans(
            n_clusters=self.cfg.n_clusters,
            max_iter=self.cfg.max_iter,
            tol=self.cfg.tol,
            n_init=self.cfg.n_init,
            rng=self.rng,
        )
        fitted = km.fit(X)
        distances = euclidean_distances(X, fitted.centroids, fitted.labels)
        thresholds = self._thresholds(distances, fitted.labels)

        mapping, method, diagnostics = assign_cluster_classes(
            cluster_labels=fitted.labels,
            y_poisoned=y_poisoned,
            n_clusters=self.cfg.n_clusters,
            method=self.cfg.label_assignment,
            majority_margin=self.cfg.majority_margin,
            trusted_index=trusted_index,
            trusted_y=trusted_y,
        )

        # Relabel only the samples inside their cluster's mean-distance shell.
        within = distances < thresholds[fitted.labels]
        cluster_class = np.array([mapping[int(k)] for k in fitted.labels], dtype=int)
        y_repaired = y_poisoned.copy()
        y_repaired[within] = cluster_class[within]

        # Trusted anchors are known-good; never let the defence overwrite them.
        trusted_mask = np.zeros(y_poisoned.size, dtype=bool)
        if trusted_index is not None and trusted_y is not None and len(trusted_index) > 0:
            idx = np.asarray(trusted_index, dtype=int)
            y_repaired[idx] = np.asarray(trusted_y, dtype=int)
            trusted_mask[idx] = True

        keep_mask, sample_weight = self._outer_policy(within, trusted_mask)

        return DefenceResult(
            name=self.name,
            y_repaired=y_repaired,
            y_input=y_poisoned,
            cluster_labels=fitted.labels,
            centroids=fitted.centroids,
            distances=distances,
            thresholds=thresholds,
            cluster_to_class=mapping,
            relabel_mask=within,
            assignment_method=method,
            keep_mask=keep_mask,
            sample_weight=sample_weight,
            metadata={
                "inertia": fitted.inertia,
                "kmeans_iterations": fitted.n_iter,
                "threshold_scope": self.cfg.threshold_scope,
                "outer_policy": self.cfg.outer_policy,
                "within_threshold_fraction": float(within.mean()),
                **diagnostics,
            },
        )

    def _outer_policy(
        self,
        within: np.ndarray,
        trusted_mask: np.ndarray,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Decide what happens to the samples KCD did not relabel.

        ``keep``
            Published rule: beyond-mu samples stay in D'' with whatever label
            the adversary left on them.
        ``drop``
            Beyond-mu samples are removed from D''.  KCD has no basis for
            trusting their labels, so this treats them as unusable rather than
            as ground truth -- ordinary data sanitisation.
        ``downweight``
            A middle ground: they stay, but contribute ``outer_weight`` of a
            normal sample to the loss.

        Trusted anchors are always kept at full weight, whichever policy is in
        force, because their labels are known-good regardless of distance.
        """
        policy = self.cfg.outer_policy
        if policy == "keep":
            return None, None
        reliable = within | trusted_mask
        if policy == "drop":
            return reliable, None
        if policy == "downweight":
            weight = np.where(reliable, 1.0, self.cfg.outer_weight)
            return None, weight
        raise ValueError("defence.outer_policy must be one of keep|drop|downweight")
