"""Cloud database (Module 1, block 4).

Persists the raw sensor data store and the labelled training set
``S = (x_i, y_i)`` that the IDS training pipeline consumes.  SQLite keeps the
project self-contained while still exercising a real store-and-query path.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

RAW_COLUMNS: tuple[str, ...] = (
    "vehicle_id",
    "sequence",
    "timestamp",
    "ingest_time",
    "link",
    "latency_ms",
    "clock_offset_ms",
    "tyre_temperature",
    "tyre_pressure",
    "latitude",
    "longitude",
    "speed",
    "fuel_level",
    "label",
    "attack_type",
    "tampered",
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS raw_readings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    vehicle_id      INTEGER NOT NULL,
    sequence        INTEGER NOT NULL,
    timestamp       REAL    NOT NULL,
    ingest_time     REAL,
    link            TEXT,
    latency_ms      REAL,
    clock_offset_ms REAL,
    tyre_temperature REAL   NOT NULL,
    tyre_pressure   REAL    NOT NULL,
    latitude        REAL    NOT NULL,
    longitude       REAL    NOT NULL,
    speed           REAL,
    fuel_level      REAL,
    label           INTEGER NOT NULL,
    attack_type     TEXT    NOT NULL DEFAULT 'none',
    tampered        INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_raw_vehicle ON raw_readings(vehicle_id, sequence);
CREATE INDEX IF NOT EXISTS idx_raw_label   ON raw_readings(label);
"""


@dataclass
class LabelledDataset:
    """The dataset ``S = (x_i, y_i)`` handed to the training pipeline."""

    X: np.ndarray
    y: np.ndarray
    feature_names: tuple[str, ...]
    attack_types: np.ndarray | None = None
    vehicle_ids: np.ndarray | None = None

    def __len__(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])

    def class_counts(self) -> dict[int, int]:
        values, counts = np.unique(self.y, return_counts=True)
        return {int(v): int(c) for v, c in zip(values, counts)}

    def summary(self) -> dict:
        counts = self.class_counts()
        n = len(self)
        return {
            "n_samples": n,
            "n_features": self.n_features,
            "feature_names": list(self.feature_names),
            "class_counts": counts,
            "malicious_ratio": counts.get(1, 0) / n if n else 0.0,
        }


class CloudDatabase:
    """Raw sensor data store plus labelled-training-set extraction."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = str(path) if path is not None else ":memory:"
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def __enter__(self) -> "CloudDatabase":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self.conn.close()

    def reset(self) -> None:
        """Drop every stored reading (used when re-running an experiment)."""
        self.conn.execute("DELETE FROM raw_readings")
        self.conn.commit()

    def store(self, records: Iterable[dict]) -> int:
        """Insert ingested records; returns the number of rows written."""
        rows = [tuple(rec.get(col) for col in RAW_COLUMNS) for rec in records]
        if not rows:
            return 0
        placeholders = ",".join("?" * len(RAW_COLUMNS))
        self.conn.executemany(
            f"INSERT INTO raw_readings ({','.join(RAW_COLUMNS)}) VALUES ({placeholders})",
            rows,
        )
        self.conn.commit()
        return len(rows)

    def count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM raw_readings").fetchone()[0])

    def labelled_dataset(self, feature_names: Sequence[str]) -> LabelledDataset:
        """Materialise ``S = (x_i, y_i)`` over the requested feature columns."""
        cols = list(feature_names)
        unknown = set(cols) - set(RAW_COLUMNS)
        if unknown:
            raise KeyError(f"unknown feature column(s): {sorted(unknown)}")
        query = (
            f"SELECT {','.join(cols)}, label, attack_type, vehicle_id "
            "FROM raw_readings ORDER BY id"
        )
        rows = self.conn.execute(query).fetchall()
        if not rows:
            raise RuntimeError("cloud database is empty; run the acquisition stage first")
        X = np.asarray([[row[c] for c in cols] for row in rows], dtype=float)
        y = np.asarray([row["label"] for row in rows], dtype=int)
        attacks = np.asarray([row["attack_type"] for row in rows], dtype=object)
        vehicles = np.asarray([row["vehicle_id"] for row in rows], dtype=int)
        return LabelledDataset(X, y, tuple(cols), attacks, vehicles)

    def attack_breakdown(self) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT attack_type, COUNT(*) AS n FROM raw_readings GROUP BY attack_type"
        ).fetchall()
        return {row["attack_type"]: int(row["n"]) for row in rows}
