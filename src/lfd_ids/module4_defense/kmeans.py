"""A small, deterministic k-means used by the defence.

Implemented here rather than pulled from scikit-learn so that clustering
behaviour is fixed by the project's own seed and cannot drift with a library
upgrade -- the defence's numbers have to be reproducible.  ``KMeans`` below
uses k-means++ seeding and Lloyd's algorithm, and is verified against
``sklearn.cluster.KMeans`` in the test suite.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class KMeansResult:
    """Fitted clustering: centroids, hard assignments and inertia."""

    centroids: np.ndarray
    labels: np.ndarray
    inertia: float
    n_iter: int


def _pairwise_sq_dists(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Squared Euclidean distance from every row of ``X`` to every centroid."""
    # ||x - c||^2 = ||x||^2 - 2 x.c + ||c||^2, clipped to stay non-negative.
    x_sq = np.einsum("ij,ij->i", X, X)[:, None]
    c_sq = np.einsum("ij,ij->i", centroids, centroids)[None, :]
    return np.maximum(x_sq - 2.0 * (X @ centroids.T) + c_sq, 0.0)


class KMeans:
    """Lloyd's algorithm with k-means++ initialisation and ``n_init`` restarts."""

    def __init__(
        self,
        n_clusters: int = 2,
        max_iter: int = 300,
        tol: float = 1e-6,
        n_init: int = 10,
        rng: np.random.Generator | None = None,
    ) -> None:
        if n_clusters < 1:
            raise ValueError("n_clusters must be >= 1")
        self.n_clusters = int(n_clusters)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.n_init = int(n_init)
        self.rng = rng or np.random.default_rng()

    def _kmeans_plusplus(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        centroids = np.empty((self.n_clusters, X.shape[1]), dtype=float)
        centroids[0] = X[self.rng.integers(0, n)]
        closest = _pairwise_sq_dists(X, centroids[:1]).ravel()
        for k in range(1, self.n_clusters):
            total = closest.sum()
            if total <= 0:
                centroids[k] = X[self.rng.integers(0, n)]
            else:
                # Sample proportionally to the squared distance from the
                # nearest already-chosen centre.
                centroids[k] = X[self.rng.choice(n, p=closest / total)]
            closest = np.minimum(closest, _pairwise_sq_dists(X, centroids[k : k + 1]).ravel())
        return centroids

    def _single_run(self, X: np.ndarray) -> KMeansResult:
        centroids = self._kmeans_plusplus(X)
        labels = np.zeros(X.shape[0], dtype=int)
        n_iter = 0
        for n_iter in range(1, self.max_iter + 1):
            dists = _pairwise_sq_dists(X, centroids)
            labels = np.argmin(dists, axis=1)
            new_centroids = centroids.copy()
            for k in range(self.n_clusters):
                members = X[labels == k]
                if members.size:
                    new_centroids[k] = members.mean(axis=0)
                else:
                    # Re-seed an emptied cluster onto the worst-fit point.
                    new_centroids[k] = X[np.argmax(dists.min(axis=1))]
            shift = float(np.sum((new_centroids - centroids) ** 2))
            centroids = new_centroids
            if shift <= self.tol:
                break
        dists = _pairwise_sq_dists(X, centroids)
        labels = np.argmin(dists, axis=1)
        inertia = float(dists[np.arange(X.shape[0]), labels].sum())
        return KMeansResult(centroids=centroids, labels=labels, inertia=inertia, n_iter=n_iter)

    def fit(self, X: np.ndarray) -> KMeansResult:
        """Run ``n_init`` restarts and keep the lowest-inertia solution."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"expected a 2-D feature matrix, got shape {X.shape}")
        if X.shape[0] < self.n_clusters:
            raise ValueError(
                f"cannot fit {self.n_clusters} clusters to {X.shape[0]} samples"
            )
        best: KMeansResult | None = None
        for _ in range(max(self.n_init, 1)):
            result = self._single_run(X)
            if best is None or result.inertia < best.inertia:
                best = result
        assert best is not None
        # Canonical ordering (by first coordinate) so cluster ids are stable
        # across runs and restarts.
        order = np.argsort(best.centroids[:, 0], kind="mergesort")
        remap = np.empty(self.n_clusters, dtype=int)
        remap[order] = np.arange(self.n_clusters)
        return KMeansResult(
            centroids=best.centroids[order],
            labels=remap[best.labels],
            inertia=best.inertia,
            n_iter=best.n_iter,
        )


def euclidean_distances(X: np.ndarray, centroids: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Distance from each sample to the centroid of its assigned cluster."""
    X = np.asarray(X, dtype=float)
    diff = X - centroids[np.asarray(labels, dtype=int)]
    return np.sqrt(np.einsum("ij,ij->i", diff, diff))
