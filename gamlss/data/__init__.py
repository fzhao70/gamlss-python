"""Datasets from the R package gamlss.data (exported to CSV)."""

from __future__ import annotations

import os

import pandas as pd

_HERE = os.path.dirname(__file__)


def load_data(name: str) -> pd.DataFrame:
    """Load one of the bundled gamlss.data datasets (e.g. "abdom")."""
    path = os.path.join(_HERE, f"{name}.csv")
    if not os.path.exists(path):
        available = sorted(
            f[:-4] for f in os.listdir(_HERE) if f.endswith(".csv")
        )
        raise ValueError(f"unknown dataset {name!r}; available: {available}")
    return pd.read_csv(path)
