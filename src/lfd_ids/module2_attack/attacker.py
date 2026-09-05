"""Attacker model (Module 2, block 3).

Describes what the adversary can see and touch, and provides the concrete
capability used by the rest of the module: a man-in-the-middle hook that can be
installed on the V2V/V2I uplink to perturb labels of packets in flight.

Threat model
------------
* **Knowledge** -- ``white_box`` means the adversary reads the labelled cloud
  dataset ``D``; ``black_box`` means it only observes the transmitted packets
  and must select flip targets without the ground-truth labels.
* **Access** -- ``mitm`` intercepts the 5G/4G uplink; ``compromised_labeler``
  models a tampered in-vehicle labelling component that mislabels at source.
* **Scope** -- ``single_vehicle`` restricts tampering to one compromised unit,
  ``fleet`` allows it across the whole fleet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..module1_acquisition.v2x import Packet
from .baglfa import BAGLFA
from .base import LabelFlippingAttack
from .bootlfa import BOOTLFA

KNOWLEDGE_LEVELS: tuple[str, ...] = ("white_box", "black_box")
ACCESS_LEVELS: tuple[str, ...] = ("mitm", "compromised_labeler")
SCOPES: tuple[str, ...] = ("single_vehicle", "fleet")

ATTACK_REGISTRY: dict[str, type[LabelFlippingAttack]] = {
    "BOOTLFA": BOOTLFA,
    "BAGLFA": BAGLFA,
}


@dataclass
class AttackerModel:
    """The adversary's capabilities, and the attack instances it can mount."""

    rng: np.random.Generator
    knowledge: str = "white_box"
    access: str = "mitm"
    scope: str = "fleet"
    target_vehicle: int = 0

    def __post_init__(self) -> None:
        if self.knowledge not in KNOWLEDGE_LEVELS:
            raise ValueError(f"knowledge must be one of {KNOWLEDGE_LEVELS}")
        if self.access not in ACCESS_LEVELS:
            raise ValueError(f"access must be one of {ACCESS_LEVELS}")
        if self.scope not in SCOPES:
            raise ValueError(f"scope must be one of {SCOPES}")

    # ------------------------------------------------------------- capability

    def mitm_hook(self, flip_probability: float) -> Callable[[Packet], Packet]:
        """Return a hook that inverts labels of intercepted packets.

        Installed on :class:`~lfd_ids.module1_acquisition.v2x.V2XChannel`, this
        is the in-flight version of the attack: it corrupts the labelled record
        before it ever reaches the cloud database.
        """

        def hook(packet: Packet) -> Packet:
            if self.scope == "single_vehicle" and packet.reading.vehicle_id != self.target_vehicle:
                return packet
            if self.rng.random() < flip_probability:
                packet.reading.label = int(abs(1 - packet.reading.label))
                packet.tampered = True
            return packet

        return hook

    # ----------------------------------------------------------------- attacks

    def build_attack(
        self,
        name: str,
        intensity: float,
        n_estimators: int = 10,
        subset_fraction: float = 0.8,
        selection: str = "random",
    ) -> LabelFlippingAttack:
        """Instantiate BOOTLFA or BAGLFA under this attacker's constraints."""
        key = name.upper()
        if key not in ATTACK_REGISTRY:
            raise ValueError(f"unknown attack '{name}'; expected one of {sorted(ATTACK_REGISTRY)}")
        if self.knowledge == "black_box" and selection != "random":
            # Without label access the adversary cannot aim its flips.
            raise ValueError(
                "a black-box attacker cannot use a label-directed selection strategy; "
                "use selection='random'"
            )
        cls = ATTACK_REGISTRY[key]
        if key == "BAGLFA":
            return cls(
                intensity=intensity,
                n_estimators=n_estimators,
                subset_fraction=subset_fraction,
                selection=selection,
                rng=self.rng,
            )
        return cls(
            intensity=intensity,
            n_estimators=n_estimators,
            selection=selection,
            rng=self.rng,
        )

    def describe(self) -> dict:
        return {
            "knowledge": self.knowledge,
            "access": self.access,
            "scope": self.scope,
            "target_vehicle": self.target_vehicle if self.scope == "single_vehicle" else None,
            "capabilities": {
                "reads_ground_truth_labels": self.knowledge == "white_box",
                "intercepts_uplink": self.access == "mitm",
                "controls_in_vehicle_labeller": self.access == "compromised_labeler",
            },
        }
