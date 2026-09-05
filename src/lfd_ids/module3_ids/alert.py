"""Alert module (Module 3, block 4).

Turns per-record IDS probabilities into fleet-level actions: it raises a
cyberattack flag once a vehicle produces ``consecutive_trigger`` malicious
classifications in a row, and it records the model-broadcast event that closes
the feedback loop back to the vehicles in the architecture diagram.

Requiring consecutive detections is what keeps a single false positive from
paging an operator, and it is why the false-negative rate matters more than
raw accuracy: a missed detection breaks the run of hits that raises the flag.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import AlertConfig


@dataclass
class Alert:
    """One raised cyberattack flag."""

    vehicle_id: int
    first_index: int
    length: int
    mean_confidence: float

    def as_dict(self) -> dict:
        return {
            "vehicle_id": self.vehicle_id,
            "first_index": self.first_index,
            "length": self.length,
            "mean_confidence": self.mean_confidence,
        }


@dataclass
class BroadcastRecord:
    """A hardened model published to the connected-vehicle fleet."""

    model_version: str
    accuracy: float
    fnr: float
    n_vehicles: int

    def as_dict(self) -> dict:
        return {
            "model_version": self.model_version,
            "accuracy": self.accuracy,
            "fnr": self.fnr,
            "n_vehicles": self.n_vehicles,
        }


@dataclass
class AlertModule:
    """Cyberattack flag trigger plus the model-broadcast channel."""

    cfg: AlertConfig
    alerts: list[Alert] = field(default_factory=list)
    broadcasts: list[BroadcastRecord] = field(default_factory=list)

    def evaluate_stream(
        self,
        scores: np.ndarray,
        vehicle_ids: np.ndarray | None = None,
    ) -> list[Alert]:
        """Scan a scored stream and raise a flag per qualifying run.

        Runs are tracked per vehicle, so interleaved fleet traffic does not
        accumulate a spurious streak across different vehicles.
        """
        scores = np.asarray(scores, dtype=float)
        if vehicle_ids is None:
            vehicle_ids = np.zeros(scores.size, dtype=int)
        vehicle_ids = np.asarray(vehicle_ids, dtype=int)
        flagged = scores >= self.cfg.decision_threshold

        raised: list[Alert] = []
        run_start: dict[int, int] = {}
        run_len: dict[int, int] = {}
        run_sum: dict[int, float] = {}
        for i, (vid, hit) in enumerate(zip(vehicle_ids, flagged)):
            vid = int(vid)
            if hit:
                if run_len.get(vid, 0) == 0:
                    run_start[vid] = i
                    run_sum[vid] = 0.0
                run_len[vid] = run_len.get(vid, 0) + 1
                run_sum[vid] += float(scores[i])
                if run_len[vid] == self.cfg.consecutive_trigger:
                    raised.append(
                        Alert(
                            vehicle_id=vid,
                            first_index=run_start[vid],
                            length=run_len[vid],
                            mean_confidence=run_sum[vid] / run_len[vid],
                        )
                    )
            else:
                run_len[vid] = 0
        self.alerts.extend(raised)
        return raised

    def broadcast(
        self,
        model_version: str,
        accuracy: float,
        fnr: float,
        n_vehicles: int,
    ) -> BroadcastRecord | None:
        """Publish a retrained model to the fleet (the deployed-model feedback)."""
        if not self.cfg.broadcast_model:
            return None
        record = BroadcastRecord(model_version, float(accuracy), float(fnr), int(n_vehicles))
        self.broadcasts.append(record)
        return record

    def summary(self) -> dict:
        return {
            "decision_threshold": self.cfg.decision_threshold,
            "consecutive_trigger": self.cfg.consecutive_trigger,
            "n_alerts": len(self.alerts),
            "alerts": [a.as_dict() for a in self.alerts[:20]],
            "broadcasts": [b.as_dict() for b in self.broadcasts],
        }
