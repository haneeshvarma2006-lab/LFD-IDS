"""Typed configuration for every stage of the LFD-IDS pipeline.

The dataclasses below mirror the four architecture modules.  A YAML file may
override any subset of the defaults; see ``configs/default.yaml``.
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Sequence

# The four features named by the architecture diagram's Preprocessing block.
CORE_FEATURES: tuple[str, ...] = (
    "tyre_temperature",
    "tyre_pressure",
    "latitude",
    "longitude",
)

# Additional sensors emitted by the vehicle sensor suite.
EXTENDED_FEATURES: tuple[str, ...] = CORE_FEATURES + ("speed", "fuel_level")


@dataclass
class AcquisitionConfig:
    """Module 1 -- connected-vehicle sensor data acquisition."""

    n_vehicles: int = 40
    samples_per_vehicle: int = 500
    malicious_ratio: float = 0.50
    sampling_hz: float = 1.0
    #: Bounding box of the simulated road network (lat_min, lat_max, lon_min, lon_max).
    region: tuple[float, float, float, float] = (12.90, 13.10, 77.55, 77.75)
    n_routes: int = 8
    ambient_temp_c: float = 28.0
    nominal_pressure_psi: float = 34.0
    sensor_noise: float = 1.0
    #: V2V/V2I channel behaviour.
    packet_loss_rate: float = 0.01
    latency_ms_mean: float = 22.0
    latency_ms_std: float = 6.0
    clock_skew_ms: float = 35.0
    #: Cloud ingestion validation.
    drop_invalid_records: bool = True
    #: Feature columns handed to the IDS.
    feature_set: Sequence[str] = CORE_FEATURES
    #: Optional path to a real CSV dataset; when set the simulator is bypassed.
    csv_path: str | None = None
    csv_label_column: str = "label"
    #: Where the cloud database is materialised (``None`` keeps it in memory).
    database_path: str | None = "results/cloud_db.sqlite"


@dataclass
class AttackConfig:
    """Module 2 -- the label-flipping poisoning engine."""

    #: Poisoning intensities p, as fractions of the training set.
    intensities: Sequence[float] = (0.30, 0.40, 0.50)
    #: Number of bootstrap resamples (BOOTLFA) / bagging subsets (BAGLFA).
    n_estimators: int = 10
    #: Size of each bagging subset relative to the training set (BAGLFA).
    subset_fraction: float = 0.8
    #: Sample-selection strategy: random | benign_to_malicious | malicious_to_benign
    #: | boundary.
    selection: str = "random"
    #: Also train the per-subset ensemble (M1..MB) and its aggregator.
    train_ensemble: bool = True
    #: Attacker capability model.
    attacker_knowledge: str = "white_box"  # white_box | black_box
    attacker_access: str = "mitm"  # mitm | compromised_labeler
    attacker_scope: str = "fleet"  # single_vehicle | fleet


@dataclass
class ModelConfig:
    """Module 3 -- the sequential MLP intrusion detector."""

    #: Hidden layer widths, matching Dense 64 / 32 / 16 / 8 of the architecture.
    hidden_units: Sequence[int] = (64, 32, 16, 8)
    #: Activation per hidden layer.  The abstract names tanh/relu/tanh for the
    #: last three; the input Dense-64 layer's activation is configurable.
    activations: Sequence[str] = ("relu", "tanh", "relu", "tanh")
    output_activation: str = "sigmoid"
    learning_rate: float = 1e-3
    epochs: int = 200
    batch_size: int = 64
    l2: float = 0.0
    #: Stop when validation loss has not improved for this many epochs
    #: (``0`` disables early stopping).
    early_stopping_patience: int = 0
    #: numpy | keras | auto
    backend: str = "auto"
    verbose: int = 0


@dataclass
class PreprocessConfig:
    """Module 3 -- preprocessing (train/val/test split and normalisation)."""

    train_frac: float = 0.60
    val_frac: float = 0.20
    test_frac: float = 0.20
    #: minmax | standard
    scaler: str = "minmax"
    stratify: bool = True


@dataclass
class DefenceConfig:
    """Module 4 -- the K-means clustering defence."""

    n_clusters: int = 2
    max_iter: int = 300
    tol: float = 1e-6
    n_init: int = 10
    #: How a cluster is mapped onto a class label:
    #: majority (vote of poisoned labels) | anchor (small trusted set) |
    #: auto (majority when its margin is decisive, else anchor).
    label_assignment: str = "auto"
    #: Margin below which ``auto`` falls back to the trusted anchor set.
    majority_margin: float = 0.10
    #: Fraction of the training set the defender holds out as trusted.
    trusted_fraction: float = 0.05
    #: per_cluster | global -- how the mean distance threshold mu is estimated.
    threshold_scope: str = "per_cluster"
    #: Multiplier applied to mu (1.0 reproduces the published rule d < mu).
    threshold_scale: float = 1.0
    #: What to do with samples beyond mu, whose labels KCD does not trust
    #: enough to rewrite.  "drop" removes them from D'' and is the setting that
    #: restores near-baseline accuracy; "keep" is the literal relabel-only rule
    #: (kept as an ablation); "downweight" is the middle ground.
    outer_policy: str = "drop"
    #: Weight given to beyond-mu samples when outer_policy == "downweight".
    outer_weight: float = 0.25


@dataclass
class AlertConfig:
    """Module 3 -- the alert module."""

    decision_threshold: float = 0.5
    #: Consecutive malicious classifications required to raise a fleet alert.
    consecutive_trigger: int = 3
    broadcast_model: bool = True


@dataclass
class ExperimentConfig:
    """Top-level configuration for a full attack-and-defence experiment."""

    seed: int = 42
    output_dir: str = "results"
    attacks: Sequence[str] = ("BOOTLFA", "BAGLFA")
    #: Repeat the whole matrix with these seed offsets and report mean +/- std.
    n_repeats: int = 1
    save_figures: bool = True
    acquisition: AcquisitionConfig = field(default_factory=AcquisitionConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    defence: DefenceConfig = field(default_factory=DefenceConfig)
    alert: AlertConfig = field(default_factory=AlertConfig)


def _coerce(cls: type, payload: dict[str, Any]) -> Any:
    """Build ``cls`` from ``payload``, recursing into nested dataclass fields."""
    kwargs: dict[str, Any] = {}
    known = {f.name: f for f in fields(cls)}
    for key, value in payload.items():
        if key not in known:
            raise KeyError(f"unknown configuration key '{key}' for {cls.__name__}")
        field_type = known[key].type
        default = known[key].default
        default_factory = known[key].default_factory  # type: ignore[attr-defined]
        nested = None
        if is_dataclass(field_type) and isinstance(field_type, type):
            nested = field_type
        elif default_factory is not None and default_factory is not MISSING:
            candidate = default_factory()
            if is_dataclass(candidate):
                nested = type(candidate)
        elif is_dataclass(default):
            nested = type(default)
        if nested is not None and isinstance(value, dict):
            kwargs[key] = _coerce(nested, value)
        elif isinstance(value, list):
            kwargs[key] = tuple(value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def load_config(path: str | Path | None = None, **overrides: Any) -> ExperimentConfig:
    """Load an :class:`ExperimentConfig`, optionally from YAML, with overrides.

    ``overrides`` accepts dotted keys such as ``model.epochs=50``.
    """
    payload: dict[str, Any] = {}
    if path is not None:
        import yaml

        text = Path(path).read_text(encoding="utf-8")
        payload = yaml.safe_load(text) or {}
    cfg = _coerce(ExperimentConfig, payload)
    for dotted, value in overrides.items():
        apply_override(cfg, dotted, value)
    validate(cfg)
    return cfg


def apply_override(cfg: Any, dotted: str, value: Any) -> None:
    """Set ``cfg.a.b.c = value`` from the dotted key ``"a.b.c"``."""
    parts = dotted.split(".")
    target = cfg
    for part in parts[:-1]:
        if not hasattr(target, part):
            raise KeyError(f"unknown configuration section '{part}' in '{dotted}'")
        target = getattr(target, part)
    leaf = parts[-1]
    if not hasattr(target, leaf):
        raise KeyError(f"unknown configuration key '{dotted}'")
    setattr(target, leaf, value)


#: Dotted paths of every field that must hold a sequence.  A scalar arriving
#: from YAML or ``--set`` (``attack.intensities=0.4``) is widened to a 1-tuple.
SEQUENCE_FIELDS: tuple[str, ...] = (
    "attacks",
    "attack.intensities",
    "acquisition.feature_set",
    "acquisition.region",
    "model.hidden_units",
    "model.activations",
)


def normalise(cfg: ExperimentConfig) -> None:
    """Coerce sequence-valued fields that were given as bare scalars."""
    for dotted in SEQUENCE_FIELDS:
        target = cfg
        *path, leaf = dotted.split(".")
        for part in path:
            target = getattr(target, part)
        value = getattr(target, leaf)
        if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
            setattr(target, leaf, (value,))
        elif isinstance(value, list):
            setattr(target, leaf, tuple(value))


def validate(cfg: ExperimentConfig) -> None:
    """Normalise sequence fields, then reject a self-inconsistent configuration."""
    normalise(cfg)
    pre = cfg.preprocess
    total = pre.train_frac + pre.val_frac + pre.test_frac
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"train/val/test fractions must sum to 1.0, got {total}")
    if not 0.0 < cfg.acquisition.malicious_ratio < 1.0:
        raise ValueError("acquisition.malicious_ratio must lie in (0, 1)")
    for p in cfg.attack.intensities:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"poisoning intensity must lie in [0, 1], got {p}")
    if len(cfg.model.hidden_units) != len(cfg.model.activations):
        raise ValueError("model.hidden_units and model.activations must be the same length")
    if cfg.defence.n_clusters < 2:
        raise ValueError("defence.n_clusters must be at least 2")
    unknown = set(cfg.attacks) - {"BOOTLFA", "BAGLFA"}
    if unknown:
        raise ValueError(f"unknown attack(s): {sorted(unknown)}")
    if cfg.model.backend not in {"auto", "numpy", "keras"}:
        raise ValueError("model.backend must be one of auto|numpy|keras")
    if cfg.defence.label_assignment not in {"majority", "anchor", "auto"}:
        raise ValueError("defence.label_assignment must be one of majority|anchor|auto")
    if cfg.defence.outer_policy not in {"keep", "drop", "downweight"}:
        raise ValueError("defence.outer_policy must be one of keep|drop|downweight")
    if cfg.defence.threshold_scope not in {"per_cluster", "global"}:
        raise ValueError("defence.threshold_scope must be one of per_cluster|global")
    if not 0.0 <= cfg.defence.outer_weight <= 1.0:
        raise ValueError("defence.outer_weight must lie in [0, 1]")
