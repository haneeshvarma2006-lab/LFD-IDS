"""LFD-IDS: Label-Flipping Data poisoning attacks and defence for
cloud-based Intrusion Detection Systems in Connected Vehicles.

The package is organised as the four modules of the system architecture:

* :mod:`lfd_ids.module1_acquisition` -- connected-vehicle sensor data
  acquisition (sensor suite, V2V/V2I channel, cloud ingestion, cloud database).
* :mod:`lfd_ids.module2_attack` -- the label-flipping poisoning engine
  (attacker model, ``BOOTLFA`` and ``BAGLFA``).
* :mod:`lfd_ids.module3_ids` -- the deep-learning IDS (preprocessing, the
  sequential MLP classifier, the evaluation engine and the alert module).
* :mod:`lfd_ids.module4_defense` -- the K-means clustering defence
  (``KCD`` and its ``BOOTKCD`` / ``BAGKCD`` variants, retraining and
  deployment).
"""

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "config",
    "pipeline",
    "module1_acquisition",
    "module2_attack",
    "module3_ids",
    "module4_defense",
]
