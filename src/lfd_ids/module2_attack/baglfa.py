"""BAGLFA -- bagging-based label-flipping attack (Module 2).

The adversary generates ``B`` bagging subsets (each a fraction ``f`` of the
training set, drawn without replacement) and inverts labels *across* those
subsets rather than within a single resampled view.  The flip budget is filled
round-robin over the subsets, so every subset contributes an equal share of the
corruption.  That even spread is the "variance-reduced label noise" of the
architecture: unlike BOOTLFA, no region of the training distribution absorbs a
disproportionate part of the budget.

As with BOOTLFA the global budget is ``round(p * N)`` labels, keeping the two
attacks comparable at each poisoning intensity.  One caveat is inherent to the
method: BAGLFA can only invert labels it can reach through a bagging subset, so
a sample that lands in none of the ``B`` subsets is untouchable.  With the
defaults that is a fraction ``(1 - f)^B`` of the training set -- about one
sample in 10^7 at ``f = 0.8, B = 10`` -- but it means the realised flip count
can fall a little short of the budget for small ``B`` or small ``f``.  The
shortfall is reported as ``uncovered_fraction`` in the result metadata.
"""

from __future__ import annotations

import numpy as np

from .base import (
    LabelFlippingAttack,
    PoisonResult,
    SubsetPoison,
    invert_labels,
    selection_weights,
)


class BAGLFA(LabelFlippingAttack):
    """Bagging-subset label-flipping attack with a majority-voting aggregator."""

    name = "BAGLFA"

    def __init__(
        self,
        intensity: float,
        n_estimators: int = 10,
        subset_fraction: float = 0.8,
        selection: str = "random",
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(intensity, n_estimators, selection, rng)
        if not 0.0 < subset_fraction <= 1.0:
            raise ValueError(f"subset_fraction must lie in (0, 1], got {subset_fraction}")
        self.subset_fraction = float(subset_fraction)

    def poison(
        self,
        y: np.ndarray,
        surrogate_scores: np.ndarray | None = None,
    ) -> PoisonResult:
        y_clean = np.asarray(y, dtype=int)
        n = y_clean.size
        weights = selection_weights(y_clean, self.selection, surrogate_scores)
        subset_size = max(1, int(round(self.subset_fraction * n)))
        budget = self.budget(n)

        # Draw the bagging subsets first; the flip budget is then spread over
        # them round-robin so each subset carries an equal share.
        subset_indices = [
            self.rng.choice(n, size=subset_size, replace=False)
            for _ in range(self.n_estimators)
        ]
        flip_set = self._round_robin_flips(subset_indices, weights, budget)

        y_poisoned = invert_labels(y_clean, np.asarray(sorted(flip_set), dtype=int))
        flip_mask = y_poisoned != y_clean

        subsets: list[SubsetPoison] = []
        coverage = np.zeros(n, dtype=float)
        for indices in subset_indices:
            coverage[indices] += 1.0
            flipped_mask = flip_mask[indices]
            subsets.append(
                SubsetPoison(
                    indices=indices,
                    y=y_poisoned[indices].copy(),
                    flipped=flipped_mask.copy(),
                )
            )

        per_subset_rates = [float(s.flipped.mean()) for s in subsets] or [0.0]
        return PoisonResult(
            name=self.name,
            intensity=self.intensity,
            y_clean=y_clean,
            y_poisoned=y_poisoned,
            flip_mask=flip_mask,
            subsets=subsets,
            metadata={
                "n_estimators": self.n_estimators,
                "subset_fraction": self.subset_fraction,
                "subset_size": subset_size,
                "selection": self.selection,
                "budget": budget,
                "uncovered_fraction": float(np.mean(coverage == 0)),
                "per_subset_flip_rate_mean": float(np.mean(per_subset_rates)),
                "per_subset_flip_rate_std": float(np.std(per_subset_rates)),
                "noise_profile": "variance_reduced",
            },
        )

    def _round_robin_flips(
        self,
        subset_indices: list[np.ndarray],
        weights: np.ndarray,
        budget: int,
    ) -> set[int]:
        """Fill the flip budget by taking turns across the bagging subsets."""
        if budget <= 0:
            return set()
        # Pre-shuffle each subset's eligible members; drawing from the front of
        # the queue is then a uniform draw without replacement.
        queues: list[list[int]] = []
        for indices in subset_indices:
            eligible = indices[weights[indices] > 0]
            if self.selection == "boundary":
                # Highest-weight (most ambiguous) samples first.
                eligible = eligible[np.argsort(-weights[eligible])]
            else:
                eligible = self.rng.permutation(eligible)
            queues.append(list(eligible))

        flip_set: set[int] = set()
        exhausted = 0
        cursor = 0
        while len(flip_set) < budget and exhausted < len(queues):
            queue = queues[cursor % len(queues)]
            cursor += 1
            if not queue:
                exhausted += 1
                continue
            exhausted = 0
            while queue:
                candidate = int(queue.pop(0))
                if candidate not in flip_set:
                    flip_set.add(candidate)
                    break
        return flip_set
