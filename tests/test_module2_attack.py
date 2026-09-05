"""Module 2: the fraction-p selector, label inversion, BOOTLFA and BAGLFA."""

from __future__ import annotations

import numpy as np
import pytest

from lfd_ids.module2_attack import (
    BAGLFA,
    BOOTLFA,
    AttackerModel,
    invert_labels,
    selection_weights,
)


@pytest.fixture
def labels():
    rng = np.random.default_rng(11)
    return (rng.random(1500) < 0.5).astype(int)


def test_invert_labels_matches_the_published_operator():
    y = np.array([0, 1, 0, 1])
    out = invert_labels(y, np.array([0, 3]))
    np.testing.assert_array_equal(out, [1, 1, 0, 0])
    # y' = |1 - y| on the selected positions, and the input is not mutated.
    np.testing.assert_array_equal(y, [0, 1, 0, 1])


def test_selection_weights_restrict_the_flip_direction():
    y = np.array([0, 1, 0, 1])
    np.testing.assert_array_equal(selection_weights(y, "random"), np.ones(4))
    np.testing.assert_array_equal(selection_weights(y, "benign_to_malicious"), [1, 0, 1, 0])
    np.testing.assert_array_equal(selection_weights(y, "malicious_to_benign"), [0, 1, 0, 1])
    boundary = selection_weights(y, "boundary", np.array([0.5, 0.99, 0.01, 0.5]))
    assert boundary[0] > boundary[1] and boundary[0] > boundary[2]
    with pytest.raises(ValueError, match="surrogate scores"):
        selection_weights(y, "boundary")
    with pytest.raises(ValueError, match="unknown selection strategy"):
        selection_weights(y, "nope")


@pytest.mark.parametrize("intensity", [0.0, 0.3, 0.4, 0.5, 1.0])
@pytest.mark.parametrize("attack_cls", [BOOTLFA, BAGLFA])
def test_realised_intensity_matches_the_request(labels, attack_cls, intensity):
    result = attack_cls(intensity, n_estimators=5, rng=np.random.default_rng(1)).poison(labels)
    budget = round(intensity * labels.size)
    # BAGLFA can only reach samples that land in some bagging subset, so its
    # flip count may fall marginally short of the budget; BOOTLFA always hits it.
    uncovered = result.metadata.get("uncovered_fraction", 0.0)
    assert budget - round(uncovered * labels.size) - 1 <= result.n_flipped <= budget
    assert result.realised_intensity == pytest.approx(intensity, abs=1e-3)
    # Only labels in the flip set moved, and each moved by inversion.
    changed = result.y_poisoned != result.y_clean
    np.testing.assert_array_equal(
        result.y_poisoned[changed], 1 - result.y_clean[changed]
    )


def test_baglfa_cannot_flip_samples_no_subset_covers(labels):
    """A sample in none of the B subsets is out of BAGLFA's reach, by design."""
    result = BAGLFA(
        1.0, n_estimators=2, subset_fraction=0.5, rng=np.random.default_rng(21)
    ).poison(labels)
    uncovered = result.metadata["uncovered_fraction"]
    assert uncovered > 0.0
    assert result.n_flipped == pytest.approx(
        labels.size * (1.0 - uncovered), abs=1
    )


@pytest.mark.parametrize("attack_cls", [BOOTLFA, BAGLFA])
def test_directed_selection_only_flips_one_direction(labels, attack_cls):
    result = attack_cls(
        0.3, n_estimators=5, selection="malicious_to_benign", rng=np.random.default_rng(2)
    ).poison(labels)
    directions = result.flip_direction_counts()
    assert directions["benign_to_malicious"] == 0
    assert directions["malicious_to_benign"] == result.n_flipped


def test_bootlfa_builds_bootstrap_subsets(labels):
    result = BOOTLFA(0.4, n_estimators=6, rng=np.random.default_rng(3)).poison(labels)
    assert len(result.subsets) == 6
    for subset in result.subsets:
        # A bootstrap resample is N draws with replacement, so it has the same
        # length as the training set and repeats some rows.
        assert subset.indices.size == labels.size
        assert np.unique(subset.indices).size < labels.size
        assert subset.flipped.sum() == pytest.approx(0.4 * labels.size, rel=0.02)
    # Across 6 resamples almost every row gets drawn at least once.
    assert result.metadata["never_resampled_fraction"] < 0.01
    assert result.metadata["noise_profile"] == "distribution_level"


def test_baglfa_builds_disjoint_sized_subsets_and_spreads_noise(labels):
    result = BAGLFA(
        0.4, n_estimators=6, subset_fraction=0.5, rng=np.random.default_rng(4)
    ).poison(labels)
    assert len(result.subsets) == 6
    for subset in result.subsets:
        assert subset.indices.size == labels.size // 2
        # Sampling without replacement: no row appears twice in a subset.
        assert np.unique(subset.indices).size == subset.indices.size
    # The point of BAGLFA: the per-subset flip rate barely varies.
    assert result.metadata["per_subset_flip_rate_std"] < 0.05
    assert result.metadata["noise_profile"] == "variance_reduced"


def test_attacks_select_different_samples_at_the_same_intensity(labels):
    boot = BOOTLFA(0.4, rng=np.random.default_rng(5)).poison(labels)
    bag = BAGLFA(0.4, rng=np.random.default_rng(5)).poison(labels)
    assert boot.n_flipped == bag.n_flipped
    assert not np.array_equal(boot.flip_mask, bag.flip_mask)


def test_attacks_are_reproducible_under_a_seed(labels):
    a = BOOTLFA(0.4, rng=np.random.default_rng(7)).poison(labels)
    b = BOOTLFA(0.4, rng=np.random.default_rng(7)).poison(labels)
    np.testing.assert_array_equal(a.y_poisoned, b.y_poisoned)


def test_attacker_model_capability_checks():
    rng = np.random.default_rng(8)
    attacker = AttackerModel(rng, knowledge="black_box")
    with pytest.raises(ValueError, match="black-box attacker cannot"):
        attacker.build_attack("BOOTLFA", 0.3, selection="boundary")
    assert attacker.build_attack("baglfa", 0.3).name == "BAGLFA"
    with pytest.raises(ValueError, match="unknown attack"):
        attacker.build_attack("NOPE", 0.3)
    with pytest.raises(ValueError, match="knowledge must be"):
        AttackerModel(rng, knowledge="psychic")


def test_mitm_hook_respects_single_vehicle_scope():
    from lfd_ids.module1_acquisition.sensors import SensorReading
    from lfd_ids.module1_acquisition.v2x import Packet

    attacker = AttackerModel(
        np.random.default_rng(9), scope="single_vehicle", target_vehicle=2
    )
    hook = attacker.mitm_hook(1.0)
    targeted = Packet(SensorReading(2, 0, 0.0, 50, 34, 12.9, 77.6, 40, 90, 0), "5G", 0, 0, 1)
    other = Packet(SensorReading(3, 0, 0.0, 50, 34, 12.9, 77.6, 40, 90, 0), "5G", 0, 0, 1)
    assert hook(targeted).reading.label == 1 and targeted.tampered
    assert hook(other).reading.label == 0 and not other.tampered


def test_invalid_parameters_are_rejected():
    with pytest.raises(ValueError, match="intensity must lie"):
        BOOTLFA(1.5)
    with pytest.raises(ValueError, match="n_estimators"):
        BOOTLFA(0.3, n_estimators=0)
    with pytest.raises(ValueError, match="subset_fraction"):
        BAGLFA(0.3, subset_fraction=0.0)
