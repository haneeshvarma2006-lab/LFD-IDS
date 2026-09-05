"""Module 4: k-means, the KCD relabelling rule, its variants and deployment."""

from __future__ import annotations

import numpy as np
import pytest

from lfd_ids.config import DefenceConfig
from lfd_ids.module2_attack import BAGLFA, BOOTLFA
from lfd_ids.module3_ids.evaluate import Metrics, evaluate, label_recovery_metrics
from lfd_ids.module4_defense import (
    BAGKCD,
    BOOTKCD,
    CloudModelPublisher,
    KMeans,
    KMeansClusteringDefence,
    RobustnessValidator,
    assign_cluster_classes,
    build_defence,
    euclidean_distances,
    select_trusted_anchors,
)


@pytest.fixture
def blobs():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0.0, 0.6, (400, 4)), rng.normal(4.0, 0.6, (400, 4))])
    y = np.array([0] * 400 + [1] * 400)
    return X, y


# ------------------------------------------------------------------ k-means


def test_kmeans_matches_sklearn_inertia(blobs):
    cluster = pytest.importorskip("sklearn.cluster")
    X, _ = blobs
    mine = KMeans(2, rng=np.random.default_rng(1)).fit(X)
    ref = cluster.KMeans(n_clusters=2, n_init=10, random_state=0).fit(X)
    assert mine.inertia == pytest.approx(ref.inertia_, rel=1e-9)
    agreement = max(
        np.mean(mine.labels == ref.labels_), np.mean(mine.labels == 1 - ref.labels_)
    )
    assert agreement == 1.0


def test_kmeans_recovers_planted_clusters(blobs):
    X, y = blobs
    result = KMeans(2, rng=np.random.default_rng(2)).fit(X)
    agreement = max(np.mean(result.labels == y), np.mean(result.labels == 1 - y))
    assert agreement > 0.99
    assert result.centroids.shape == (2, 4)


def test_kmeans_cluster_ids_are_canonical(blobs):
    """Cluster ids must not depend on the seed, or the defence is unstable."""
    X, _ = blobs
    a = KMeans(2, rng=np.random.default_rng(3)).fit(X)
    b = KMeans(2, rng=np.random.default_rng(99)).fit(X)
    np.testing.assert_array_equal(a.labels, b.labels)
    assert a.centroids[0, 0] < a.centroids[1, 0]


def test_kmeans_rejects_impossible_requests():
    with pytest.raises(ValueError, match="cannot fit"):
        KMeans(5).fit(np.zeros((3, 2)))
    with pytest.raises(ValueError, match="2-D feature matrix"):
        KMeans(2).fit(np.zeros(10))


def test_euclidean_distances_are_to_the_assigned_centroid():
    X = np.array([[0.0, 0.0], [3.0, 4.0]])
    centroids = np.array([[0.0, 0.0], [10.0, 10.0]])
    d = euclidean_distances(X, centroids, np.array([0, 0]))
    np.testing.assert_allclose(d, [0.0, 5.0])


# -------------------------------------------------------- cluster labelling


def test_majority_assignment_and_its_margin():
    clusters = np.array([0, 0, 0, 1, 1, 1])
    y = np.array([0, 0, 1, 1, 1, 0])
    mapping, method, diag = assign_cluster_classes(clusters, y, 2, "majority", 0.1)
    assert mapping == {0: 0, 1: 1}
    assert method == "majority"
    assert diag["majority_margins"] == [pytest.approx(1 / 3), pytest.approx(1 / 3)]


def test_auto_falls_back_to_anchors_on_a_tied_vote():
    clusters = np.array([0, 0, 1, 1])
    y_poisoned = np.array([0, 1, 0, 1])  # a perfect 50/50 tie in both clusters
    trusted_idx = np.array([0, 2])
    trusted_y = np.array([1, 0])
    mapping, method, diag = assign_cluster_classes(
        clusters, y_poisoned, 2, "auto", 0.1, trusted_idx, trusted_y
    )
    assert method == "anchor"
    assert mapping == {0: 1, 1: 0}
    assert diag["n_trusted"] == 2


def test_anchor_without_a_trusted_set_is_an_error():
    with pytest.raises(ValueError, match="requires a trusted subset"):
        assign_cluster_classes(np.array([0, 1]), np.array([0, 1]), 2, "anchor", 0.1)


def test_degenerate_majority_is_detected():
    # Both clusters vote "malicious" -- an unusable mapping.
    clusters = np.array([0, 0, 1, 1])
    y = np.array([1, 1, 1, 1])
    _, method, diag = assign_cluster_classes(clusters, y, 2, "auto", 0.1)
    assert diag["majority_degenerate"] is True
    assert method == "majority_fallback"


# ---------------------------------------------------------------------- KCD


def test_kcd_only_relabels_inside_the_mean_distance_shell(blobs):
    X, y = blobs
    rng = np.random.default_rng(4)
    y_poisoned = y.copy()
    flipped = rng.choice(y.size, 240, replace=False)
    y_poisoned[flipped] = 1 - y_poisoned[flipped]

    cfg = DefenceConfig(outer_policy="keep", n_init=4)
    result = KMeansClusteringDefence(cfg, np.random.default_rng(5)).defend(X, y_poisoned)
    # Every changed label sits strictly inside its cluster's threshold.
    changed = result.y_repaired != result.y_input
    assert np.all(result.relabel_mask[changed])
    assert np.all(result.distances[changed] < result.thresholds[result.cluster_labels[changed]])
    # Roughly half a cluster's mass lies within its mean radius.
    assert 0.4 < result.metadata["within_threshold_fraction"] < 0.7


def test_kcd_reduces_label_error(blobs):
    X, y = blobs
    rng = np.random.default_rng(6)
    y_poisoned = y.copy()
    flipped = rng.choice(y.size, 320, replace=False)  # p = 40%
    y_poisoned[flipped] = 1 - y_poisoned[flipped]
    result = KMeansClusteringDefence(
        DefenceConfig(n_init=4), np.random.default_rng(7)
    ).defend(X, y_poisoned)
    recovery = label_recovery_metrics(y, y_poisoned, result.y_repaired)
    # KCD can only repair flips inside the mean-distance shell, so the residual
    # label error should land near before * (1 - within_threshold_fraction).
    within = result.metadata["within_threshold_fraction"]
    expected = recovery["label_error_before"] * (1.0 - within)
    assert recovery["label_error_after"] == pytest.approx(expected, abs=0.03)
    assert recovery["correction_rate"] == pytest.approx(within, abs=0.06)


@pytest.mark.parametrize("policy,expect_drop", [("keep", False), ("drop", True), ("downweight", False)])
def test_outer_policies_shape_the_training_set(blobs, policy, expect_drop):
    X, y = blobs
    result = KMeansClusteringDefence(
        DefenceConfig(outer_policy=policy, n_init=4), np.random.default_rng(8)
    ).defend(X, y)
    X_out, y_out, weight = result.training_set(X)
    if expect_drop:
        assert X_out.shape[0] < X.shape[0]
        assert y_out.size == X_out.shape[0]
    else:
        assert X_out.shape[0] == X.shape[0]
    if policy == "downweight":
        assert weight is not None and set(np.unique(weight)) == {0.25, 1.0}
    elif policy == "keep":
        assert weight is None


def test_trusted_anchors_are_never_overwritten(blobs):
    X, y = blobs
    y_poisoned = 1 - y  # every label inverted
    trusted_idx, trusted_y = select_trusted_anchors(y, 0.1, np.random.default_rng(9))
    result = KMeansClusteringDefence(
        DefenceConfig(n_init=4), np.random.default_rng(10)
    ).defend(X, y_poisoned, trusted_idx, trusted_y)
    np.testing.assert_array_equal(result.y_repaired[trusted_idx], trusted_y)


def test_kcd_rejects_mismatched_inputs(blobs):
    X, y = blobs
    with pytest.raises(ValueError, match="rows but"):
        KMeansClusteringDefence(DefenceConfig(n_init=2)).defend(X, y[:10])


# ----------------------------------------------------------------- variants


@pytest.mark.parametrize("cls", [BOOTKCD, BAGKCD])
def test_variants_match_the_base_defence_shape(blobs, cls):
    X, y = blobs
    rng = np.random.default_rng(11)
    y_poisoned = y.copy()
    flipped = rng.choice(y.size, 320, replace=False)
    y_poisoned[flipped] = 1 - y_poisoned[flipped]
    result = cls(DefenceConfig(n_init=3), n_estimators=4, rng=np.random.default_rng(12)).defend(
        X, y_poisoned
    )
    assert result.y_repaired.shape == y.shape
    assert result.metadata["n_estimators"] == 4
    recovery = label_recovery_metrics(y, y_poisoned, result.y_repaired)
    assert recovery["label_error_after"] < recovery["label_error_before"]


def test_build_defence_registry():
    cfg = DefenceConfig(n_init=2)
    rng = np.random.default_rng(13)
    assert build_defence("KCD", cfg, rng).name == "KCD"
    assert build_defence("bootkcd", cfg, rng).name == "BOOTKCD"
    assert build_defence("BAGKCD", cfg, rng).name == "BAGKCD"
    with pytest.raises(ValueError, match="unknown defence"):
        build_defence("nope", cfg, rng)


# ------------------------------------------------- anchors, gating, publishing


def test_select_trusted_anchors_is_stratified():
    y = np.array([0] * 90 + [1] * 10)
    idx, labels = select_trusted_anchors(y, 0.1, np.random.default_rng(14))
    assert idx.size == labels.size
    # Both classes represented, so neither cluster is left without an anchor.
    assert set(np.unique(labels)) == {0, 1}
    np.testing.assert_array_equal(labels, y[idx])


def test_select_trusted_anchors_handles_edges():
    y = np.array([0, 1, 0, 1])
    idx, labels = select_trusted_anchors(y, 0.0, np.random.default_rng(15))
    assert idx.size == 0 and labels.size == 0
    idx, labels = select_trusted_anchors(y, 1.5, np.random.default_rng(15))
    assert idx.size == 4


def _metrics(accuracy: float, fnr: float, auc: float) -> Metrics:
    n = 1000
    tp = int(round((1 - fnr) * n / 2))
    fn = n // 2 - tp
    tn = int(round(accuracy * n)) - tp
    fp = n - tp - fn - tn
    y = np.concatenate([np.zeros(tn + fp, dtype=int), np.ones(tp + fn, dtype=int)])
    scores = np.concatenate(
        [np.zeros(tn), np.ones(fp), np.ones(tp), np.zeros(fn)]
    )
    m = evaluate(y, scores)
    return Metrics(accuracy, m.precision, 1 - fnr, m.f1, fnr, m.fpr, m.specificity, auc,
                   m.confusion, n)


def test_robustness_validator_gates_bad_models():
    validator = RobustnessValidator()
    baseline = _metrics(0.995, 0.005, 0.999)
    assert validator.validate(_metrics(0.99, 0.01, 0.99), baseline).passed
    weak = validator.validate(_metrics(0.80, 0.30, 0.85), baseline)
    assert not weak.passed
    assert len(weak.reasons) >= 3  # accuracy floor, FNR cap, AUC floor
    drifted = validator.validate(_metrics(0.93, 0.02, 0.98), baseline)
    assert not drifted.passed
    assert any("below the clean baseline" in r for r in drifted.reasons)


def test_publisher_deploys_only_models_that_pass():
    validator = RobustnessValidator()
    publisher = CloudModelPublisher()
    good = _metrics(0.99, 0.01, 0.995)
    bad = _metrics(0.55, 0.45, 0.60)
    publisher.publish("v1", good, "clean", validator.validate(good))
    publisher.publish("v2", bad, "poisoned", validator.validate(bad))
    assert publisher.current is not None and publisher.current.version == "v1"
    assert len(publisher.history) == 1
    assert publisher.summary()["rejected"][0]["version"] == "v2"


# ------------------------------------------------------- attack <-> defence


@pytest.mark.parametrize(
    "attack_cls,defence_name", [(BOOTLFA, "BOOTKCD"), (BAGLFA, "BAGKCD")]
)
def test_defence_undoes_a_real_attack(split, attack_cls, defence_name):
    poison = attack_cls(0.4, n_estimators=4, rng=np.random.default_rng(16)).poison(
        split.y_train
    )
    trusted_idx, trusted_y = select_trusted_anchors(
        split.y_train, 0.05, np.random.default_rng(17)
    )
    defence = build_defence(
        defence_name, DefenceConfig(n_init=4), np.random.default_rng(18), n_estimators=4
    )
    result = defence.defend(split.X_train, poison.y_poisoned, trusted_idx, trusted_y)
    recovery = label_recovery_metrics(split.y_train, poison.y_poisoned, result.y_repaired)
    assert recovery["label_error_before"] == pytest.approx(0.4, abs=0.01)
    assert recovery["label_error_after"] < 0.25
    assert recovery["correction_rate"] > 0.4
