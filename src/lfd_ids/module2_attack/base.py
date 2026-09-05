"""Shared machinery for the label-flipping attacks (Module 2).

Both BOOTLFA and BAGLFA are built from the same two primitives named in the
architecture diagram: a *fraction-p sample selector* and a *label inversion
operator* ``y' = |1 - y|``.  They differ only in how the flip budget is spread
over the resampled views of the training set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

#: Strategies for choosing which samples the adversary inverts.
SELECTION_STRATEGIES: tuple[str, ...] = (
    "random",
    "benign_to_malicious",
    "malicious_to_benign",
    "boundary",
)


def invert_labels(y: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Apply ``y' = |1 - y|`` at ``indices`` and return a new label vector."""
    out = np.asarray(y, dtype=int).copy()
    out[indices] = np.abs(1 - out[indices])
    return out


def selection_weights(
    y: np.ndarray,
    strategy: str,
    scores: np.ndarray | None = None,
) -> np.ndarray:
    """Per-sample eligibility weights for the fraction-p selector.

    ``random`` weights every sample equally (the published attack).  The
    class-directed strategies restrict the adversary to one flip direction,
    and ``boundary`` concentrates the budget on samples a surrogate model is
    least certain about -- the strongest attack of the four.
    """
    y = np.asarray(y, dtype=int)
    if strategy == "random":
        return np.ones(y.size, dtype=float)
    if strategy == "benign_to_malicious":
        return (y == 0).astype(float)
    if strategy == "malicious_to_benign":
        return (y == 1).astype(float)
    if strategy == "boundary":
        if scores is None:
            raise ValueError("the 'boundary' strategy needs surrogate scores")
        # Highest weight where the surrogate probability is nearest 0.5.
        return 1.0 - 2.0 * np.abs(np.asarray(scores, dtype=float) - 0.5) + 1e-9
    raise ValueError(f"unknown selection strategy '{strategy}'; expected one of {SELECTION_STRATEGIES}")


def choose_indices(
    candidates: np.ndarray,
    k: int,
    weights: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw ``k`` distinct entries of ``candidates`` without replacement.

    ``weights`` is indexed by the *values* in ``candidates`` (i.e. positions in
    the full training set).  Zero-weight candidates are ineligible; if fewer
    than ``k`` remain eligible, every eligible candidate is returned.
    """
    candidates = np.asarray(candidates, dtype=int)
    if k <= 0 or candidates.size == 0:
        return np.empty(0, dtype=int)
    w = np.asarray(weights, dtype=float)[candidates]
    eligible = candidates[w > 0]
    if eligible.size == 0:
        return np.empty(0, dtype=int)
    if k >= eligible.size:
        return eligible
    p = w[w > 0]
    p = p / p.sum()
    return rng.choice(eligible, size=k, replace=False, p=p)


@dataclass
class SubsetPoison:
    """One resampled view of the training set after label inversion."""

    #: Positions in the original training set (may repeat for bootstraps).
    indices: np.ndarray
    #: Labels of ``indices`` after inversion.
    y: np.ndarray
    #: Which positions *within this subset* were inverted.
    flipped: np.ndarray

    def __len__(self) -> int:
        return int(self.indices.size)


@dataclass
class PoisonResult:
    """The output of a label-flipping attack.

    ``y_poisoned`` is the poisoned training set ``D'`` that the cloud stores
    and the IDS trains on; ``subsets`` are the per-estimator views
    ``M_1 .. M_B`` are trained from.
    """

    name: str
    intensity: float
    y_clean: np.ndarray
    y_poisoned: np.ndarray
    flip_mask: np.ndarray
    subsets: list[SubsetPoison] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def n_flipped(self) -> int:
        return int(self.flip_mask.sum())

    @property
    def realised_intensity(self) -> float:
        """Fraction of ``D'`` whose label differs from ground truth."""
        return float(self.flip_mask.mean()) if self.flip_mask.size else 0.0

    def flip_direction_counts(self) -> dict[str, int]:
        """How many benign->malicious and malicious->benign inversions landed."""
        flipped = self.flip_mask
        return {
            "benign_to_malicious": int(np.sum(flipped & (self.y_clean == 0))),
            "malicious_to_benign": int(np.sum(flipped & (self.y_clean == 1))),
        }

    def summary(self) -> dict:
        return {
            "attack": self.name,
            "intensity": self.intensity,
            "realised_intensity": self.realised_intensity,
            "n_flipped": self.n_flipped,
            "n_train": int(self.y_clean.size),
            "n_subsets": len(self.subsets),
            "flip_directions": self.flip_direction_counts(),
            **self.metadata,
        }


class LabelFlippingAttack:
    """Base class holding the parameters shared by BOOTLFA and BAGLFA."""

    name = "LFA"

    def __init__(
        self,
        intensity: float,
        n_estimators: int = 10,
        selection: str = "random",
        rng: np.random.Generator | None = None,
    ) -> None:
        if not 0.0 <= intensity <= 1.0:
            raise ValueError(f"intensity must lie in [0, 1], got {intensity}")
        if selection not in SELECTION_STRATEGIES:
            raise ValueError(
                f"unknown selection strategy '{selection}'; expected one of {SELECTION_STRATEGIES}"
            )
        if n_estimators < 1:
            raise ValueError("n_estimators must be >= 1")
        self.intensity = float(intensity)
        self.n_estimators = int(n_estimators)
        self.selection = selection
        self.rng = rng or np.random.default_rng()

    def budget(self, n: int) -> int:
        """Global number of labels the adversary is allowed to invert."""
        return int(round(self.intensity * n))

    def poison(
        self,
        y: np.ndarray,
        surrogate_scores: np.ndarray | None = None,
    ) -> PoisonResult:  # pragma: no cover - implemented by subclasses
        raise NotImplementedError
