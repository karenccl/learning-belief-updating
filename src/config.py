"""Configuration loading and project paths."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "default_config.yaml"
RESULTS = ROOT / "results"
FITS = RESULTS / "fits"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
DATA_SIM = ROOT / "data" / "simulated"


def load_config(path: Path | str = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)
