"""Module 3: preprocessing, the MLP, the metric set and the alert module."""

from __future__ import annotations

import numpy as np
import pytest

from lfd_ids.config import ModelConfig
from lfd_ids.module3_ids import (
    AlertModule,
    NumpyMLP,
    Scaler,
    build_model,
    describe_architecture,
    evaluate,
    keras_available,
    label_recovery_metrics,
    per_attack_recall,
    preprocess,
    roc_auc,
    stratified_split,
)
from lfd_ids.module3_ids.evaluate import ConfusionMatrix


# ------------------------------------------------------------- preprocessing


def test_split_is_60_20_20_and_stratified(split):
    total = split.y_train.size + split.y_val.size + split.y_test.size
    assert split.y_train.size / total == pytest.approx(0.60, abs=0.01)
    assert split.y_val.size / total == pytest.approx(0.20, abs=0.01)
    assert split.y_test.size / total == pytest.approx(0.20, abs=0.01)
    for part in (split.y_train, split.y_val, split.y_test):
        assert part.mean() == pytest.approx(split.y_train.mean(), abs=0.02)


def test_splits_are_disjoint_and_cover_the_dataset(split):
    joined = np.concatenate([split.train_index, split.val_index, split.test_index])
    assert np.unique(joined).size == joined.size


def test_scaler_is_fitted_on_train_only(split):
    # Min-max normalisation puts the training split in [0, 1] exactly; the
    # other splits may sit slightly outside, which proves no leakage.
    assert split.X_train.min() == pytest.approx(0.0, abs=1e-9)
    assert split.X_train.max() == pytest.approx(1.0, abs=1e-9)


def test_scaler_roundtrip():
    rng = np.random.default_rng(0)
    X = rng.normal(3.0, 2.0, (100, 4))
    for kind in ("minmax", "standard"):
        scaler = Scaler.fit(X, kind)
        np.testing.assert_allclose(scaler.inverse_transform(scaler.transform(X)), X)
    with pytest.raises(ValueError, match="unknown scaler"):
        Scaler.fit(X, "quantile")


def test_scaler_handles_a_constant_feature():
    X = np.column_stack([np.ones(20), np.arange(20.0)])
    out = Scaler.fit(X, "minmax").transform(X)
    assert np.all(np.isfinite(out))


def test_stratified_split_preserves_class_ratio():
    y = np.array([0] * 80 + [1] * 20)
    groups = stratified_split(y, (0.6, 0.2, 0.2), np.random.default_rng(0))
    assert sum(g.size for g in groups) == 100
    for g in groups:
        assert y[g].mean() == pytest.approx(0.2, abs=0.02)


def test_with_train_labels_rejects_a_length_mismatch(split):
    with pytest.raises(ValueError, match="expected"):
        split.with_train_labels(np.zeros(3, dtype=int))


# -------------------------------------------------------------------- model


def test_architecture_matches_the_specification():
    cfg = ModelConfig()
    assert describe_architecture(cfg, 4) == [
        "Input(4)",
        "Dense(64, relu)",
        "Dense(32, tanh)",
        "Dense(16, relu)",
        "Dense(8, tanh)",
        "Dense(1, sigmoid)",
    ]


def test_numpy_mlp_parameter_count():
    # (4*64+64) + (64*32+32) + (32*16+16) + (16*8+8) + (8*1+1)
    expected = 320 + 2080 + 528 + 136 + 9
    assert NumpyMLP(4, ModelConfig(), seed=0).n_parameters() == expected


def test_numpy_mlp_learns_a_nonlinear_boundary():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1600, 4))
    y = ((X[:, 0] ** 2 + X[:, 1] ** 2) > 1.5).astype(int)
    cfg = ModelConfig(epochs=120, backend="numpy")
    model = NumpyMLP(4, cfg, seed=0).fit(X[:1200], y[:1200], X[1200:], y[1200:])
    assert np.mean(model.predict(X[1200:]) == y[1200:]) > 0.93
    # Loss actually decreased rather than wandering.
    assert model.history["loss"][-1] < model.history["loss"][0] * 0.5


def test_numpy_mlp_is_deterministic_under_a_seed():
    rng = np.random.default_rng(1)
    X, y = rng.normal(size=(300, 4)), (rng.random(300) < 0.5).astype(int)
    cfg = ModelConfig(epochs=10, backend="numpy")
    a = NumpyMLP(4, cfg, seed=3).fit(X, y).predict_proba(X)
    b = NumpyMLP(4, cfg, seed=3).fit(X, y).predict_proba(X)
    np.testing.assert_allclose(a, b)


def test_sample_weights_shift_the_decision():
    rng = np.random.default_rng(2)
    X = np.linspace(-2, 2, 400).reshape(-1, 1)
    y = (X.ravel() > 0).astype(int)
    y_noisy = y.copy()
    y_noisy[:100] = 1  # a corrupted block on the negative side
    weight = np.ones(400)
    weight[:100] = 0.0  # ignore it entirely
    cfg = ModelConfig(epochs=150, backend="numpy", hidden_units=(16, 8), activations=("relu", "tanh"))
    weighted = NumpyMLP(1, cfg, seed=0).fit(X, y_noisy, sample_weight=weight)
    unweighted = NumpyMLP(1, cfg, seed=0).fit(X, y_noisy)
    assert np.mean(weighted.predict(X) == y) > np.mean(unweighted.predict(X) == y)


def test_backend_resolution_and_build(small_config):
    from lfd_ids.module3_ids.model import resolve_backend

    assert resolve_backend("auto") == "numpy"
    assert resolve_backend("numpy") == "numpy"
    model = build_model(small_config.model, 4, 0)
    assert model.backend == "numpy"


@pytest.mark.skipif(not keras_available(), reason="TensorFlow not installed")
def test_keras_backend_matches_the_numpy_parameter_count():
    cfg = ModelConfig(epochs=1, backend="keras")
    assert build_model(cfg, 4, 0).n_parameters() == NumpyMLP(4, ModelConfig()).n_parameters()


# ------------------------------------------------------------------ metrics


def test_confusion_matrix_counts():
    cm = ConfusionMatrix.from_predictions(
        np.array([0, 0, 1, 1, 1]), np.array([0, 1, 1, 1, 0])
    )
    assert (cm.tn, cm.fp, cm.fn, cm.tp) == (1, 1, 1, 2)
    assert cm.total == 5
    np.testing.assert_array_equal(cm.as_array(), [[1, 1], [1, 2]])


def test_metrics_are_internally_consistent():
    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.4).astype(int)
    scores = np.clip(0.35 * y + rng.normal(0.3, 0.2, 400), 0, 1)
    m = evaluate(y, scores)
    cm = m.confusion
    assert m.accuracy == pytest.approx((cm.tp + cm.tn) / cm.total)
    assert m.recall == pytest.approx(cm.tp / (cm.tp + cm.fn))
    assert m.fnr == pytest.approx(1.0 - m.recall)
    assert m.specificity == pytest.approx(1.0 - m.fpr)
    assert m.f1 == pytest.approx(
        2 * m.precision * m.recall / (m.precision + m.recall)
    )


def test_roc_auc_matches_sklearn():
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(5)
    for _ in range(5):
        y = (rng.random(300) < 0.45).astype(int)
        scores = np.round(np.clip(0.4 * y + rng.normal(0.3, 0.25, 300), 0, 1), 2)
        assert roc_auc(y, scores) == pytest.approx(
            sklearn_metrics.roc_auc_score(y, scores)
        )


def test_roc_auc_edge_cases():
    assert np.isnan(roc_auc(np.zeros(10, dtype=int), np.random.random(10)))
    y = np.array([0, 0, 1, 1])
    assert roc_auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert roc_auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == pytest.approx(0.0)
    # All-tied scores give a coin-flip AUC.
    assert roc_auc(y, np.full(4, 0.5)) == pytest.approx(0.5)


def test_metrics_do_not_divide_by_zero():
    y = np.array([0, 0, 0, 0])
    m = evaluate(y, np.zeros(4))
    assert m.accuracy == 1.0 and m.precision == 0.0 and m.recall == 0.0 and m.fnr == 0.0


def test_per_attack_recall_breakdown():
    y = np.array([1, 1, 1, 0])
    pred = np.array([1, 0, 1, 0])
    types = np.array(["gps_spoofing", "gps_spoofing", "fuzzing", "none"], dtype=object)
    out = per_attack_recall(y, pred, types)
    assert out["gps_spoofing"] == {"n": 2, "detected": 1, "recall": 0.5}
    assert "none" not in out


def test_label_recovery_metrics_accounting():
    truth = np.array([0, 0, 1, 1, 1])
    poisoned = np.array([1, 0, 0, 0, 1])  # 3 flips
    repaired = np.array([0, 1, 1, 0, 1])  # fixed 2, missed 1, broke 1
    out = label_recovery_metrics(truth, poisoned, repaired)
    assert out["n_flipped_by_attack"] == 3
    assert out["corrected"] == 2
    assert out["missed"] == 1
    assert out["newly_corrupted"] == 1
    assert out["correction_rate"] == pytest.approx(2 / 3)


# -------------------------------------------------------------------- alert


def test_alert_needs_consecutive_detections(small_config):
    module = AlertModule(small_config.alert)
    # Two hits, a miss, then three hits: only the second run should fire.
    scores = np.array([0.9, 0.9, 0.1, 0.9, 0.9, 0.9])
    alerts = module.evaluate_stream(scores, np.zeros(6, dtype=int))
    assert len(alerts) == 1
    assert alerts[0].first_index == 3


def test_alert_runs_are_tracked_per_vehicle(small_config):
    module = AlertModule(small_config.alert)
    # Three hits interleaved across two vehicles must not fire an alert.
    scores = np.array([0.9, 0.9, 0.9, 0.9])
    vehicles = np.array([0, 1, 0, 1])
    assert module.evaluate_stream(scores, vehicles) == []


def test_broadcast_records_deployment(small_config):
    module = AlertModule(small_config.alert)
    record = module.broadcast("ids-v2", 0.99, 0.01, 40)
    assert record is not None and module.summary()["broadcasts"][0]["model_version"] == "ids-v2"
