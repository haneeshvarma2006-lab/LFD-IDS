"""BOOTLFA -- bootstrapping-based label-flipping attack (Module 2).

The adversary draws ``B`` bootstrap resamples of the training set (size ``N``,
with replacement) and inverts a fraction ``p`` of the rows of each resample.
Because a bootstrap draw over-represents some samples and omits others
(~36.8% of the training set never appears in a given resample), the flip votes
accumulated across the ``B`` resamples are *distributed* unevenly over the
training set.  BOOTLFA keeps the samples that collected the most votes, so the
corruption clusters on whatever the bootstrap distribution over-samples -- the
"distribution-level label noise" the architecture calls for.

The global flip budget is held at exactly ``round(p * N)`` so that BOOTLFA and
BAGLFA are directly comparable at each poisoning intensity; the two attacks
differ in *which* labels are inverted, not how many.
"""

from __future__ import annotations

import numpy as np

from .base import (
    LabelFlippingAttack,
    PoisonResult,
    SubsetPoison,
    choose_indices,
    invert_labels,
    selection_weights,
)


class BOOTLFA(LabelFlippingAttack):
    """Bootstrap-resampling label-flipping attack."""

    name = "BOOTLFA"

    def poison(
        self,
        y: np.ndarray,
        surrogate_scores: np.ndarray | None = None,
    ) -> PoisonResult:
        y_clean = np.asarray(y, dtype=int)
        n = y_clean.size
        weights = selection_weights(y_clean, self.selection, surrogate_scores)
        per_subset_k = int(round(self.intensity * n))

        votes = np.zeros(n, dtype=float)
        occurrences = np.zeros(n, dtype=float)
        subsets: list[SubsetPoison] = []

        for _ in range(self.n_estimators):
            # Bootstrap resample: N draws with replacement.
            draw = self.rng.integers(0, n, size=n)
            np.add.at(occurrences, draw, 1.0)
            # Select fraction p of the *positions* of this resample, so a
            # sample drawn k times gets k chances to be picked.
            positions = np.arange(draw.size)
            pos_weights = weights[draw]
            eligible = positions[pos_weights > 0]
            k = min(per_subset_k, eligible.size)
            if k > 0:
                prob = pos_weights[pos_weights > 0]
                prob = prob / prob.sum()
                picked_positions = self.rng.choice(eligible, size=k, replace=False, p=prob)
            else:
                picked_positions = np.empty(0, dtype=int)
            np.add.at(votes, draw[picked_positions], 1.0)

            # The per-bootstrap view M_b is trained on, kept for the ensemble.
            y_sub = y_clean[draw].copy()
            y_sub[picked_positions] = np.abs(1 - y_sub[picked_positions])
            flipped_mask = np.zeros(draw.size, dtype=bool)
            flipped_mask[picked_positions] = True
            subsets.append(SubsetPoison(indices=draw, y=y_sub, flipped=flipped_mask))

        budget = self.budget(n)
        flip_idx = self._top_voted(votes, weights, budget)
        y_poisoned = invert_labels(y_clean, flip_idx)
        flip_mask = y_poisoned != y_clean

        never_sampled = float(np.mean(occurrences == 0))
        return PoisonResult(
            name=self.name,
            intensity=self.intensity,
            y_clean=y_clean,
            y_poisoned=y_poisoned,
            flip_mask=flip_mask,
            subsets=subsets,
            metadata={
                "n_estimators": self.n_estimators,
                "selection": self.selection,
                "budget": budget,
                "mean_bootstrap_occurrences": float(occurrences.mean() / self.n_estimators),
                "never_resampled_fraction": never_sampled,
                "noise_profile": "distribution_level",
            },
        )

    def _top_voted(
        self,
        votes: np.ndarray,
        weights: np.ndarray,
        budget: int,
    ) -> np.ndarray:
        """Take the ``budget`` samples with the most flip votes.

        Ties are broken uniformly at random so the attack stays stochastic, and
        ineligible samples (zero selection weight) are never chosen.
        """
        if budget <= 0:
            return np.empty(0, dtype=int)
        eligible = np.flatnonzero(weights > 0)
        if eligible.size <= budget:
            return eligible
        jitter = self.rng.random(eligible.size)
        order = np.lexsort((jitter, -votes[eligible]))
        chosen = eligible[order[:budget]]
        # A budget larger than the number of vote-carrying samples is topped up
        # at random from the remaining eligible pool.
        if np.count_nonzero(votes[chosen]) < chosen.size:
            voted = chosen[votes[chosen] > 0]
            remaining = np.setdiff1d(eligible, voted, assume_unique=False)
            top_up = choose_indices(
                remaining, budget - voted.size, weights, self.rng
            )
            chosen = np.concatenate([voted, top_up])
        return np.sort(chosen)
