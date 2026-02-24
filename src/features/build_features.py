"""
Feature engineering helpers for the MTPL dataset.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_log_density(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``LogDensity = log(Density)`` (avoids skew in raw Density)."""
    out = df.copy()
    out["LogDensity"] = np.log1p(out["Density"])
    return out


def add_vehage_bin(df: pd.DataFrame) -> pd.DataFrame:
    """Bin VehAge into actuarial-style groups."""
    bins = [0, 1, 5, 10, 20, 100]
    labels = ["0-1", "2-5", "6-10", "11-20", "21+"]
    out = df.copy()
    out["VehAgeBin"] = pd.cut(
        out["VehAge"], bins=bins, labels=labels, right=True, include_lowest=True
    )
    return out


def add_drivage_bin(df: pd.DataFrame) -> pd.DataFrame:
    """Bin DrivAge into actuarial-style groups."""
    bins = [17, 25, 35, 45, 55, 65, 100]
    labels = ["18-25", "26-35", "36-45", "46-55", "56-65", "66+"]
    out = df.copy()
    out["DrivAgeBin"] = pd.cut(
        out["DrivAge"], bins=bins, labels=labels, right=True, include_lowest=True
    )
    return out


def add_bonusmalus_bin(df: pd.DataFrame) -> pd.DataFrame:
    """Bin BonusMalus into groups."""
    bins = [0, 50, 75, 100, 150, 350]
    labels = ["50", "51-75", "76-100", "101-150", "151+"]
    out = df.copy()
    out["BonusMalusBin"] = pd.cut(
        out["BonusMalus"], bins=bins, labels=labels, right=True, include_lowest=True
    )
    return out


def add_claim_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary ``HasClaim`` indicator (1 if ClaimNb > 0)."""
    out = df.copy()
    out["HasClaim"] = (out["ClaimNb"] > 0).astype(int)
    return out


def build_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature-engineering steps in sequence."""
    out = df.copy()
    out = add_log_density(out)
    out = add_vehage_bin(out)
    out = add_drivage_bin(out)
    out = add_bonusmalus_bin(out)
    out = add_claim_flag(out)
    return out
