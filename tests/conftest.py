"""Shared fixtures: a small, fast dataset and split used across the tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lfd_ids.config import load_config  # noqa: E402
from lfd_ids.module1_acquisition import acquire  # noqa: E402
from lfd_ids.module3_ids import preprocess  # noqa: E402


@pytest.fixture(scope="session")
def small_config():
    cfg = load_config()
    cfg.acquisition.n_vehicles = 8
    cfg.acquisition.samples_per_vehicle = 150
    cfg.acquisition.database_path = None
    cfg.model.backend = "numpy"
    cfg.model.epochs = 30
    cfg.attack.n_estimators = 4
    cfg.attack.train_ensemble = False
    cfg.defence.n_init = 4
    cfg.save_figures = False
    return cfg


@pytest.fixture(scope="session")
def acquisition(small_config):
    return acquire(small_config.acquisition, np.random.default_rng(0))


@pytest.fixture(scope="session")
def split(small_config, acquisition):
    return preprocess(acquisition.dataset, small_config.preprocess, np.random.default_rng(0))
