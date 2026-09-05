"""Vehicle sensor suite (Module 1, block 1).

Simulates the tyre-temperature, tyre-pressure, GPS, speed and fuel sensors of a
connected vehicle and the cyberattacks that corrupt their reported values.

Benign readings obey the physics a real tyre obeys: pressure follows the ideal
gas law with respect to tyre temperature, temperature rises with sustained
speed, fuel falls monotonically with distance travelled, and the vehicle stays
inside its fleet's operating region.  Each attack type breaks one or more of
those relationships, which is exactly the signal the IDS learns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Sequence

import numpy as np

#: Standard atmospheric pressure in psi, used to convert gauge <-> absolute.
ATMOSPHERIC_PSI = 14.696
#: Celsius -> Kelvin offset.
KELVIN = 273.15

BENIGN = 0
MALICIOUS = 1

ATTACK_TYPES: tuple[str, ...] = (
    "tpms_deflation_spoof",
    "thermal_injection",
    "gps_spoofing",
    "replay_freeze",
    "fuzzing",
)


@dataclass
class SensorReading:
    """One timestamped record produced by a single vehicle."""

    vehicle_id: int
    sequence: int
    timestamp: float
    tyre_temperature: float
    tyre_pressure: float
    latitude: float
    longitude: float
    speed: float
    fuel_level: float
    label: int
    attack_type: str = "none"

    def as_dict(self) -> dict:
        return {
            "vehicle_id": self.vehicle_id,
            "sequence": self.sequence,
            "timestamp": self.timestamp,
            "tyre_temperature": self.tyre_temperature,
            "tyre_pressure": self.tyre_pressure,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "speed": self.speed,
            "fuel_level": self.fuel_level,
            "label": self.label,
            "attack_type": self.attack_type,
        }


def pressure_from_temperature(
    temperature_c: np.ndarray | float,
    nominal_psi: float,
    reference_temp_c: float,
) -> np.ndarray | float:
    """Gauge pressure implied by the ideal gas law at constant volume.

    ``nominal_psi`` is the gauge pressure measured at ``reference_temp_c``.
    """
    p_abs_ref = nominal_psi + ATMOSPHERIC_PSI
    ratio = (np.asarray(temperature_c, dtype=float) + KELVIN) / (reference_temp_c + KELVIN)
    return p_abs_ref * ratio - ATMOSPHERIC_PSI


@dataclass
class RoadNetwork:
    """A small set of polyline routes inside the fleet's operating region."""

    region: tuple[float, float, float, float]
    n_routes: int
    rng: np.random.Generator
    routes: list[np.ndarray] = field(default_factory=list)

    def __post_init__(self) -> None:
        lat_min, lat_max, lon_min, lon_max = self.region
        for _ in range(self.n_routes):
            n_nodes = int(self.rng.integers(4, 8))
            lats = self.rng.uniform(lat_min, lat_max, n_nodes)
            lons = self.rng.uniform(lon_min, lon_max, n_nodes)
            self.routes.append(np.column_stack([lats, lons]))

    def sample_positions(self, route_idx: int, n: int, lateral_noise: float) -> np.ndarray:
        """Walk ``n`` points along a route, with small lateral GPS noise."""
        route = self.routes[route_idx % len(self.routes)]
        # Cumulative arc length so points are spaced evenly along the polyline.
        segments = np.diff(route, axis=0)
        seg_len = np.linalg.norm(segments, axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg_len)])
        total = cum[-1] if cum[-1] > 0 else 1.0
        # Loop back and forth so long journeys stay on the same route.
        t = np.linspace(0.0, 1.0, n) * 2.0
        t = np.where(t > 1.0, 2.0 - t, t) * total
        lats = np.interp(t, cum, route[:, 0])
        lons = np.interp(t, cum, route[:, 1])
        lats += self.rng.normal(0.0, lateral_noise, n)
        lons += self.rng.normal(0.0, lateral_noise, n)
        return np.column_stack([lats, lons])


class VehicleSensorSuite:
    """Generates the raw telemetry stream of one connected vehicle.

    Parameters
    ----------
    rng:
        Seeded generator; every draw comes from it so runs are reproducible.
    region:
        ``(lat_min, lat_max, lon_min, lon_max)`` operating box of the fleet.
    ambient_temp_c, nominal_pressure_psi:
        Environment and tyre baseline used by the benign physics model.
    noise:
        Multiplier on all sensor noise terms.
    stealth_fraction:
        Share of malicious readings interpolated back towards the benign
        envelope, which keeps the classes from being perfectly separable.
    """

    def __init__(
        self,
        rng: np.random.Generator,
        region: tuple[float, float, float, float] = (12.90, 13.10, 77.55, 77.75),
        ambient_temp_c: float = 28.0,
        nominal_pressure_psi: float = 34.0,
        noise: float = 1.0,
        stealth_fraction: float = 0.06,
        road_network: RoadNetwork | None = None,
    ) -> None:
        self.rng = rng
        self.region = region
        self.ambient_temp_c = ambient_temp_c
        self.nominal_pressure_psi = nominal_pressure_psi
        self.noise = noise
        self.stealth_fraction = stealth_fraction
        self.roads = road_network or RoadNetwork(region, 8, rng)

    # ---------------------------------------------------------------- benign

    def _speed_profile(self, n: int) -> np.ndarray:
        """Ornstein-Uhlenbeck style speed trace clipped to a legal range."""
        speed = np.empty(n)
        v = float(self.rng.uniform(20.0, 70.0))
        target = float(self.rng.uniform(35.0, 85.0))
        for i in range(n):
            if self.rng.random() < 0.01:
                target = float(self.rng.uniform(0.0, 100.0))
            v += 0.08 * (target - v) + self.rng.normal(0.0, 1.6 * self.noise)
            speed[i] = min(max(v, 0.0), 120.0)
        return speed

    def _benign_block(self, vehicle_id: int, n: int, route_idx: int) -> dict[str, np.ndarray]:
        speed = self._speed_profile(n)
        # Tyre temperature: ambient plus friction heating, with thermal inertia.
        heating = np.empty(n)
        h = 0.0
        for i in range(n):
            h += 0.06 * (0.34 * speed[i] - h)
            heating[i] = h
        temp = (
            self.ambient_temp_c
            + heating
            + self.rng.normal(0.0, 0.8 * self.noise, n)
        )
        # Pressure follows the ideal gas law, plus a slow per-vehicle offset for
        # tyre-to-tyre variation and a small measurement error.
        tyre_offset = self.rng.normal(0.0, 0.7)
        pressure = pressure_from_temperature(
            temp, self.nominal_pressure_psi + tyre_offset, self.ambient_temp_c
        ) + self.rng.normal(0.0, 0.25 * self.noise, n)
        positions = self.roads.sample_positions(route_idx, n, 0.0015 * self.noise)
        # Fuel drains with distance covered.
        consumed = np.cumsum(speed) / max(np.sum(speed), 1.0)
        fuel = 100.0 - consumed * float(self.rng.uniform(25.0, 60.0))
        return {
            "speed": speed,
            "tyre_temperature": temp,
            "tyre_pressure": pressure,
            "latitude": positions[:, 0],
            "longitude": positions[:, 1],
            "fuel_level": np.clip(fuel + self.rng.normal(0.0, 0.4 * self.noise, n), 0.0, 100.0),
        }

    # ------------------------------------------------------------- malicious

    def _apply_attack(self, block: dict[str, np.ndarray], attack: str, idx: np.ndarray) -> None:
        """Overwrite the readings at ``idx`` with attack-crafted values."""
        n = idx.size
        if n == 0:
            return
        rng = self.rng
        lat_min, lat_max, lon_min, lon_max = self.region
        lat_span, lon_span = lat_max - lat_min, lon_max - lon_min

        if attack == "tpms_deflation_spoof":
            # Forge a rapid-deflation alarm: pressure far below the envelope
            # while temperature keeps climbing, which the gas law forbids.
            block["tyre_pressure"][idx] = rng.uniform(11.0, 24.0, n)
            block["tyre_temperature"][idx] += rng.uniform(18.0, 40.0, n)
        elif attack == "thermal_injection":
            # Inject implausible tyre temperatures with a decoupled pressure.
            block["tyre_temperature"][idx] = rng.uniform(96.0, 138.0, n)
            block["tyre_pressure"][idx] = rng.uniform(14.0, 27.0, n)
        elif attack == "gps_spoofing":
            # Displace the reported position well outside the operating region
            # and pair it with the anomalous TPMS envelope the ECU reports
            # once the spoofed fix desynchronises the sensor fusion stack.
            angle = rng.uniform(0.0, 2.0 * np.pi, n)
            radius = rng.uniform(1.6, 4.5, n)
            block["latitude"][idx] += radius * lat_span * np.sin(angle)
            block["longitude"][idx] += radius * lon_span * np.cos(angle)
            block["tyre_temperature"][idx] += rng.uniform(20.0, 45.0, n)
            block["tyre_pressure"][idx] = rng.uniform(13.0, 26.0, n)
        elif attack == "replay_freeze":
            # Replay a stale frame: the readings stop tracking the vehicle.
            block["tyre_temperature"][idx] = rng.uniform(100.0, 125.0, n)
            block["tyre_pressure"][idx] = rng.uniform(15.0, 23.0, n)
            block["speed"][idx] = rng.uniform(0.0, 3.0, n)
            block["fuel_level"][idx] = rng.uniform(0.0, 4.0, n)
        elif attack == "fuzzing":
            # Fuzzed CAN frames drive the sensors towards their rails.
            block["tyre_temperature"][idx] = rng.uniform(88.0, 150.0, n)
            block["tyre_pressure"][idx] = rng.uniform(9.0, 25.0, n)
            block["speed"][idx] = rng.choice([0.0, 255.0], n)
            block["fuel_level"][idx] = rng.choice([0.0, 100.0], n)
        else:  # pragma: no cover - guarded by ATTACK_TYPES
            raise ValueError(f"unknown attack type: {attack}")

    def _apply_stealth(self, block: dict[str, np.ndarray], benign: dict[str, np.ndarray],
                       idx: np.ndarray) -> None:
        """Blend a subset of attacks back towards benign values.

        A stealthy adversary keeps the forged reading close to the legitimate
        one so it survives naive range checks; this is what stops the two
        classes from being trivially separable.
        """
        if idx.size == 0:
            return
        alpha = self.rng.uniform(0.72, 0.97, idx.size)
        for key in ("tyre_temperature", "tyre_pressure", "latitude", "longitude",
                    "speed", "fuel_level"):
            block[key][idx] = alpha * benign[key][idx] + (1.0 - alpha) * block[key][idx]

    # ------------------------------------------------------------------ API

    def stream(
        self,
        vehicle_id: int,
        n_samples: int,
        malicious_ratio: float,
        sampling_hz: float = 1.0,
        start_time: float = 0.0,
        attack_types: Sequence[str] = ATTACK_TYPES,
    ) -> Iterator[SensorReading]:
        """Yield ``n_samples`` readings for one vehicle, some of them attacked."""
        route_idx = int(self.rng.integers(0, max(len(self.roads.routes), 1)))
        benign = self._benign_block(vehicle_id, n_samples, route_idx)
        block = {k: v.copy() for k, v in benign.items()}

        labels = np.zeros(n_samples, dtype=int)
        attack_names = np.array(["none"] * n_samples, dtype=object)
        n_mal = int(round(malicious_ratio * n_samples))
        if n_mal > 0:
            # Attacks arrive in bursts, as a real intrusion does.
            mal_idx = self._burst_indices(n_samples, n_mal)
            labels[mal_idx] = MALICIOUS
            chosen = self.rng.choice(np.asarray(attack_types, dtype=object), mal_idx.size)
            for attack in np.unique(chosen):
                sel = mal_idx[chosen == attack]
                self._apply_attack(block, str(attack), sel)
                attack_names[sel] = attack
            n_stealth = int(round(self.stealth_fraction * mal_idx.size))
            if n_stealth > 0:
                stealth_idx = self.rng.choice(mal_idx, n_stealth, replace=False)
                self._apply_stealth(block, benign, stealth_idx)

        dt = 1.0 / sampling_hz if sampling_hz > 0 else 1.0
        for i in range(n_samples):
            yield SensorReading(
                vehicle_id=vehicle_id,
                sequence=i,
                timestamp=start_time + i * dt,
                tyre_temperature=float(block["tyre_temperature"][i]),
                tyre_pressure=float(block["tyre_pressure"][i]),
                latitude=float(block["latitude"][i]),
                longitude=float(block["longitude"][i]),
                speed=float(block["speed"][i]),
                fuel_level=float(block["fuel_level"][i]),
                label=int(labels[i]),
                attack_type=str(attack_names[i]),
            )

    def _burst_indices(self, n_samples: int, n_mal: int) -> np.ndarray:
        """Pick ``n_mal`` indices grouped into a handful of contiguous bursts."""
        if n_mal >= n_samples:
            return np.arange(n_samples)
        n_bursts = max(1, int(self.rng.integers(2, 6)))
        sizes = np.full(n_bursts, n_mal // n_bursts)
        sizes[: n_mal % n_bursts] += 1
        chosen: set[int] = set()
        for size in sizes:
            if size <= 0:
                continue
            start = int(self.rng.integers(0, max(n_samples - size, 1)))
            chosen.update(range(start, min(start + int(size), n_samples)))
        # Bursts may overlap; top up (or trim) to hit the requested count.
        pool = [i for i in range(n_samples) if i not in chosen]
        self.rng.shuffle(pool)
        while len(chosen) < n_mal and pool:
            chosen.add(pool.pop())
        out = np.array(sorted(chosen)[:n_mal], dtype=int)
        return out
