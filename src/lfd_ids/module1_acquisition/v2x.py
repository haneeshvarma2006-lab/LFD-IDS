"""V2V / V2I communication layer (Module 1, block 2).

Carries sensor readings from the vehicle to the cloud over the simulated
5G/4G link.  The channel models packet loss, latency jitter and clock skew,
and exposes a man-in-the-middle hook that the Module 2 attacker model uses to
tamper with packets in flight.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np

from .sensors import SensorReading

#: Radio access technologies a connected vehicle may attach to.
LINK_TYPES: tuple[str, ...] = ("gNodeB_5G", "eNodeB_4G", "V2V_DSRC", "Satellite_GPS")


@dataclass
class Packet:
    """A transmitted sensor frame plus its transport metadata."""

    reading: SensorReading
    link: str
    tx_time: float
    rx_time: float
    latency_ms: float
    tampered: bool = False

    @property
    def clock_offset_ms(self) -> float:
        """Skew between the vehicle clock and the ingestion clock."""
        return (self.rx_time - self.tx_time) * 1000.0 - self.latency_ms


@dataclass
class ChannelStats:
    """Counters describing one transmission run."""

    transmitted: int = 0
    delivered: int = 0
    dropped: int = 0
    tampered: int = 0
    latencies_ms: list[float] = field(default_factory=list)

    def summary(self) -> dict:
        lat = np.asarray(self.latencies_ms) if self.latencies_ms else np.zeros(1)
        return {
            "transmitted": self.transmitted,
            "delivered": self.delivered,
            "dropped": self.dropped,
            "tampered": self.tampered,
            "loss_rate": self.dropped / self.transmitted if self.transmitted else 0.0,
            "latency_ms_mean": float(lat.mean()),
            "latency_ms_p95": float(np.percentile(lat, 95)),
        }


class V2XChannel:
    """Simulated vehicle-to-everything uplink to the cloud IDS."""

    def __init__(
        self,
        rng: np.random.Generator,
        packet_loss_rate: float = 0.01,
        latency_ms_mean: float = 22.0,
        latency_ms_std: float = 6.0,
        clock_skew_ms: float = 35.0,
        links: Sequence[str] = LINK_TYPES,
    ) -> None:
        self.rng = rng
        self.packet_loss_rate = packet_loss_rate
        self.latency_ms_mean = latency_ms_mean
        self.latency_ms_std = latency_ms_std
        self.clock_skew_ms = clock_skew_ms
        self.links = tuple(links)
        self.stats = ChannelStats()
        #: Optional callback ``(Packet) -> Packet`` installed by an attacker.
        self.mitm_hook: Callable[[Packet], Packet] | None = None

    def transmit(self, readings: Iterable[SensorReading]) -> list[Packet]:
        """Send readings uplink, returning the packets that survived the hop."""
        delivered: list[Packet] = []
        for reading in readings:
            self.stats.transmitted += 1
            if self.rng.random() < self.packet_loss_rate:
                self.stats.dropped += 1
                continue
            latency = max(
                1.0, float(self.rng.normal(self.latency_ms_mean, self.latency_ms_std))
            )
            skew = float(self.rng.normal(0.0, self.clock_skew_ms))
            packet = Packet(
                reading=reading,
                link=str(self.rng.choice(self.links)),
                tx_time=reading.timestamp,
                rx_time=reading.timestamp + (latency + skew) / 1000.0,
                latency_ms=latency,
            )
            if self.mitm_hook is not None:
                packet = self.mitm_hook(packet)
                if packet.tampered:
                    self.stats.tampered += 1
            self.stats.delivered += 1
            self.stats.latencies_ms.append(latency)
            delivered.append(packet)
        return delivered
