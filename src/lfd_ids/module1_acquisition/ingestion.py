"""Cloud ingestion layer (Module 1, block 3).

Validates the format and physical plausibility of each received packet,
re-synchronises vehicle clocks and aggregates surviving packets into the
structured feature records that the cloud database stores.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .v2x import Packet

#: Hard sensor limits.  A reading outside these cannot be produced by working
#: hardware, so it is a transport/format fault rather than an attack signal;
#: attacks are detected by the IDS, not by these checks.
VALID_RANGES: dict[str, tuple[float, float]] = {
    "tyre_temperature": (-60.0, 250.0),
    "tyre_pressure": (0.0, 100.0),
    "latitude": (-90.0, 90.0),
    "longitude": (-180.0, 180.0),
    "speed": (0.0, 300.0),
    "fuel_level": (0.0, 100.0),
}


@dataclass
class IngestionStats:
    """Counters for one ingestion run."""

    received: int = 0
    accepted: int = 0
    rejected_format: int = 0
    rejected_range: int = 0
    resynced: int = 0
    rejections: dict[str, int] = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "received": self.received,
            "accepted": self.accepted,
            "rejected_format": self.rejected_format,
            "rejected_range": self.rejected_range,
            "resynced": self.resynced,
            "rejections_by_field": dict(self.rejections),
        }


class CloudIngestionLayer:
    """Format validation, timestamp synchronisation and record aggregation."""

    def __init__(self, drop_invalid: bool = True, max_clock_offset_ms: float = 250.0) -> None:
        self.drop_invalid = drop_invalid
        self.max_clock_offset_ms = max_clock_offset_ms
        self.stats = IngestionStats()

    def _validate(self, packet: Packet) -> str | None:
        """Return the name of the first failing field, or ``None`` if valid."""
        reading = packet.reading
        for name, (low, high) in VALID_RANGES.items():
            value = getattr(reading, name)
            if value is None or not (low <= float(value) <= high):
                return name
        return None

    def ingest(self, packets: list[Packet]) -> list[dict]:
        """Turn delivered packets into structured, clock-aligned records."""
        records: list[dict] = []
        for packet in packets:
            self.stats.received += 1
            bad_field = self._validate(packet)
            if bad_field is not None:
                self.stats.rejected_range += 1
                self.stats.rejections[bad_field] = self.stats.rejections.get(bad_field, 0) + 1
                if self.drop_invalid:
                    continue
            record = packet.reading.as_dict()
            # Timestamp sync: re-anchor the vehicle clock onto the cloud clock
            # by removing the measured one-way latency from the arrival time.
            offset_ms = packet.clock_offset_ms
            if abs(offset_ms) > 1e-9:
                self.stats.resynced += 1
            record["ingest_time"] = packet.rx_time - packet.latency_ms / 1000.0
            record["link"] = packet.link
            record["latency_ms"] = packet.latency_ms
            record["clock_offset_ms"] = offset_ms
            record["tampered"] = bool(packet.tampered)
            records.append(record)
            self.stats.accepted += 1
        return records
