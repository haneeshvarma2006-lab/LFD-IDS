"""End-to-end wiring of Module 1.

``acquire`` drives the full chain of the architecture's first module:

    sensor suite -> V2V/V2I channel -> cloud ingestion -> cloud database
                 -> labelled dataset S = (x_i, y_i)

``load_csv_dataset`` is the escape hatch for running the rest of the project
on a real connected-vehicle capture instead of the simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from ..config import AcquisitionConfig
from ..utils import get_logger
from .database import CloudDatabase, LabelledDataset
from .ingestion import CloudIngestionLayer
from .sensors import RoadNetwork, VehicleSensorSuite
from .v2x import V2XChannel

LOGGER = get_logger(__name__)


@dataclass
class AcquisitionResult:
    """The labelled dataset plus telemetry about how it was collected."""

    dataset: LabelledDataset
    channel_stats: dict
    ingestion_stats: dict
    attack_breakdown: dict
    source: str

    def summary(self) -> dict:
        return {
            "source": self.source,
            "dataset": self.dataset.summary(),
            "channel": self.channel_stats,
            "ingestion": self.ingestion_stats,
            "attack_breakdown": self.attack_breakdown,
        }


def acquire(cfg: AcquisitionConfig, rng: np.random.Generator) -> AcquisitionResult:
    """Run Module 1 and return the labelled training set."""
    if cfg.csv_path:
        dataset = load_csv_dataset(cfg.csv_path, cfg.feature_set, cfg.csv_label_column)
        LOGGER.info("Loaded %d records from %s", len(dataset), cfg.csv_path)
        return AcquisitionResult(
            dataset=dataset,
            channel_stats={},
            ingestion_stats={},
            attack_breakdown={},
            source=str(cfg.csv_path),
        )

    roads = RoadNetwork(tuple(cfg.region), cfg.n_routes, rng)
    suite = VehicleSensorSuite(
        rng=rng,
        region=tuple(cfg.region),
        ambient_temp_c=cfg.ambient_temp_c,
        nominal_pressure_psi=cfg.nominal_pressure_psi,
        noise=cfg.sensor_noise,
        road_network=roads,
    )
    channel = V2XChannel(
        rng=rng,
        packet_loss_rate=cfg.packet_loss_rate,
        latency_ms_mean=cfg.latency_ms_mean,
        latency_ms_std=cfg.latency_ms_std,
        clock_skew_ms=cfg.clock_skew_ms,
    )
    ingestion = CloudIngestionLayer(drop_invalid=cfg.drop_invalid_records)

    db_path = cfg.database_path
    database = CloudDatabase(db_path)
    database.reset()

    for vehicle_id in range(cfg.n_vehicles):
        readings = suite.stream(
            vehicle_id=vehicle_id,
            n_samples=cfg.samples_per_vehicle,
            malicious_ratio=cfg.malicious_ratio,
            sampling_hz=cfg.sampling_hz,
            start_time=float(rng.uniform(0.0, 3600.0)),
        )
        packets = channel.transmit(readings)
        records = ingestion.ingest(packets)
        database.store(records)

    dataset = database.labelled_dataset(cfg.feature_set)
    breakdown = database.attack_breakdown()
    database.close()

    LOGGER.info(
        "Module 1 collected %d records (%.1f%% malicious) across %d vehicles",
        len(dataset),
        100.0 * dataset.summary()["malicious_ratio"],
        cfg.n_vehicles,
    )
    return AcquisitionResult(
        dataset=dataset,
        channel_stats=channel.stats.summary(),
        ingestion_stats=ingestion.stats.summary(),
        attack_breakdown=breakdown,
        source="simulated_fleet",
    )


def load_csv_dataset(
    path: str | Path,
    feature_names: Sequence[str],
    label_column: str = "label",
) -> LabelledDataset:
    """Load a real capture, mapping its columns onto the IDS feature vector.

    The CSV must contain every name in ``feature_names`` plus ``label_column``.
    Labels may be 0/1 or any two values, in which case the lexicographically
    larger one is treated as ``malicious``.
    """
    import pandas as pd

    frame = pd.read_csv(path)
    missing = [c for c in list(feature_names) + [label_column] if c not in frame.columns]
    if missing:
        raise KeyError(
            f"{path} is missing required column(s): {missing}. "
            f"Available columns: {list(frame.columns)}"
        )
    X = frame[list(feature_names)].to_numpy(dtype=float)
    raw_labels = frame[label_column].to_numpy()
    uniques = np.unique(raw_labels)
    if uniques.size != 2:
        raise ValueError(
            f"expected a binary label column, found {uniques.size} distinct values"
        )
    positive = sorted(uniques.tolist())[-1]
    y = (raw_labels == positive).astype(int)
    attacks = (
        frame["attack_type"].to_numpy(dtype=object)
        if "attack_type" in frame.columns
        else np.where(y == 1, "malicious", "none").astype(object)
    )
    vehicles = (
        frame["vehicle_id"].to_numpy(dtype=int)
        if "vehicle_id" in frame.columns
        else np.zeros(len(y), dtype=int)
    )
    return LabelledDataset(X, y, tuple(feature_names), attacks, vehicles)
