"""Shared helpers: deterministic seeding, logging, timing and JSON I/O."""

from __future__ import annotations

import json
import logging
import os
import random
import time
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

LOGGER_NAME = "lfd_ids"


def get_logger(name: str = LOGGER_NAME) -> logging.Logger:
    """Return the package logger, configuring a stream handler once."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)-7s %(name)s | %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def set_global_seed(seed: int) -> np.random.Generator:
    """Seed every RNG the pipeline can touch and return a NumPy generator.

    TensorFlow is seeded only if it has already been imported, so that the
    NumPy backend never pays the cost of importing it.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    import sys

    if "tensorflow" in sys.modules:  # pragma: no cover - backend dependent
        sys.modules["tensorflow"].keras.utils.set_random_seed(seed)
    return np.random.default_rng(seed)


@contextmanager
def timer(label: str, logger: logging.Logger | None = None) -> Iterator[dict]:
    """Time a block and expose the elapsed seconds through the yielded dict."""
    logger = logger or get_logger()
    record: dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield record
    finally:
        record["seconds"] = time.perf_counter() - start
        logger.debug("%s took %.3fs", label, record["seconds"])


def _jsonable(obj: Any) -> Any:
    """Recursively convert dataclasses, NumPy scalars and arrays to JSON types."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return _jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def write_json(path: str | Path, payload: Any, indent: int = 2) -> Path:
    """Serialise ``payload`` to ``path``, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=indent) + "\n", encoding="utf-8")
    return path


def read_json(path: str | Path) -> Any:
    """Load JSON from ``path``."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def ensure_dir(path: str | Path) -> Path:
    """Create ``path`` (and parents) if missing and return it."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def format_pct(value: float | None, digits: int = 2) -> str:
    """Render a 0-1 ratio as a percentage string, tolerating ``None``."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    return f"{100.0 * value:.{digits}f}%"
