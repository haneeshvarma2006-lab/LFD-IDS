"""Module 1: sensor physics, the V2X channel, ingestion and the cloud store."""

from __future__ import annotations

import numpy as np
import pytest

from lfd_ids.module1_acquisition import (
    CloudDatabase,
    CloudIngestionLayer,
    V2XChannel,
    VehicleSensorSuite,
    load_csv_dataset,
)
from lfd_ids.module1_acquisition.sensors import (
    ATTACK_TYPES,
    ATMOSPHERIC_PSI,
    pressure_from_temperature,
)


def test_pressure_follows_ideal_gas_law():
    # A tyre at 34 psi gauge and 28 C gains roughly 5 psi by 60 C.
    hot = pressure_from_temperature(60.0, 34.0, 28.0)
    assert 38.0 < hot < 40.5
    # At the reference temperature the gauge reading is unchanged.
    assert pressure_from_temperature(28.0, 34.0, 28.0) == pytest.approx(34.0)
    # Absolute pressure scales linearly with absolute temperature.
    ratio = (pressure_from_temperature(100.0, 34.0, 28.0) + ATMOSPHERIC_PSI) / (
        34.0 + ATMOSPHERIC_PSI
    )
    assert ratio == pytest.approx((100.0 + 273.15) / (28.0 + 273.15))


def test_stream_hits_the_requested_malicious_ratio():
    suite = VehicleSensorSuite(np.random.default_rng(1))
    readings = list(suite.stream(3, 400, malicious_ratio=0.5))
    assert len(readings) == 400
    labels = np.array([r.label for r in readings])
    assert labels.mean() == pytest.approx(0.5, abs=0.01)
    assert {r.attack_type for r in readings if r.label == 1} <= set(ATTACK_TYPES)
    assert all(r.attack_type == "none" for r in readings if r.label == 0)


def test_attacks_break_the_temperature_pressure_coupling():
    """Attacked records must be physically distinguishable, not just relabelled."""
    suite = VehicleSensorSuite(np.random.default_rng(2), stealth_fraction=0.0)
    readings = list(suite.stream(0, 600, malicious_ratio=0.5))
    benign = np.array([[r.tyre_temperature, r.tyre_pressure] for r in readings if r.label == 0])
    mal = np.array([[r.tyre_temperature, r.tyre_pressure] for r in readings if r.label == 1])
    expected_benign = pressure_from_temperature(benign[:, 0], 34.0, 28.0)
    expected_mal = pressure_from_temperature(mal[:, 0], 34.0, 28.0)
    benign_residual = np.abs(benign[:, 1] - expected_benign).mean()
    mal_residual = np.abs(mal[:, 1] - expected_mal).mean()
    assert benign_residual < 2.0
    assert mal_residual > 10.0 * benign_residual


def test_channel_drops_packets_and_records_latency():
    suite = VehicleSensorSuite(np.random.default_rng(3))
    channel = V2XChannel(np.random.default_rng(3), packet_loss_rate=0.2)
    packets = channel.transmit(suite.stream(0, 500, 0.5))
    stats = channel.stats.summary()
    assert stats["transmitted"] == 500
    assert stats["delivered"] == len(packets)
    assert stats["loss_rate"] == pytest.approx(0.2, abs=0.05)
    assert stats["latency_ms_mean"] > 0


def test_mitm_hook_flips_labels_in_flight():
    suite = VehicleSensorSuite(np.random.default_rng(4))
    channel = V2XChannel(np.random.default_rng(4), packet_loss_rate=0.0)
    channel.mitm_hook = lambda pkt: (
        setattr(pkt.reading, "label", 1 - pkt.reading.label),
        setattr(pkt, "tampered", True),
        pkt,
    )[-1]
    packets = channel.transmit(suite.stream(0, 100, 0.5))
    assert channel.stats.tampered == len(packets) == 100


def test_ingestion_rejects_out_of_range_records():
    from lfd_ids.module1_acquisition.sensors import SensorReading
    from lfd_ids.module1_acquisition.v2x import Packet

    good = SensorReading(0, 0, 0.0, 50.0, 34.0, 12.9, 77.6, 40.0, 90.0, 0)
    bad = SensorReading(0, 1, 1.0, 50.0, 34.0, 999.0, 77.6, 40.0, 90.0, 1)
    packets = [
        Packet(good, "gNodeB_5G", 0.0, 0.02, 20.0),
        Packet(bad, "gNodeB_5G", 1.0, 1.02, 20.0),
    ]
    layer = CloudIngestionLayer(drop_invalid=True)
    records = layer.ingest(packets)
    assert len(records) == 1
    assert layer.stats.rejected_range == 1
    assert layer.stats.rejections["latitude"] == 1


def test_database_roundtrip_and_feature_selection(tmp_path):
    suite = VehicleSensorSuite(np.random.default_rng(5))
    channel = V2XChannel(np.random.default_rng(5), packet_loss_rate=0.0)
    layer = CloudIngestionLayer()
    with CloudDatabase(tmp_path / "db.sqlite") as db:
        db.store(layer.ingest(channel.transmit(suite.stream(0, 200, 0.5))))
        assert db.count() == 200
        dataset = db.labelled_dataset(("tyre_temperature", "tyre_pressure"))
        assert dataset.X.shape == (200, 2)
        assert set(np.unique(dataset.y)) == {0, 1}
        assert sum(db.attack_breakdown().values()) == 200
        with pytest.raises(KeyError):
            db.labelled_dataset(("not_a_sensor",))


def test_acquire_reports_a_consistent_pipeline(acquisition, small_config):
    summary = acquisition.summary()
    assert summary["dataset"]["n_samples"] == len(acquisition.dataset)
    assert summary["ingestion"]["accepted"] == len(acquisition.dataset)
    assert summary["channel"]["delivered"] == summary["ingestion"]["received"]
    assert summary["dataset"]["feature_names"] == list(small_config.acquisition.feature_set)


def test_load_csv_dataset(tmp_path, acquisition):
    import pandas as pd

    dataset = acquisition.dataset
    frame = pd.DataFrame(dataset.X, columns=list(dataset.feature_names))
    frame["label"] = dataset.y
    path = tmp_path / "capture.csv"
    frame.to_csv(path, index=False)

    loaded = load_csv_dataset(path, dataset.feature_names)
    assert loaded.X.shape == dataset.X.shape
    np.testing.assert_array_equal(loaded.y, dataset.y)

    with pytest.raises(KeyError, match="missing required column"):
        load_csv_dataset(path, ("nope",))
