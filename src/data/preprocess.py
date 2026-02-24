"""
Preprocessing — clean and transform raw MTPL data for modeling.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config import FREQ_PROCESSED, SEV_PROCESSED


def clean_freq(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the frequency dataset.

    Steps
    -----
    1. Drop rows with Exposure <= 0.
    2. Cap Exposure at 1 (partial-year policies).
    3. Cast IDpol to int.
    4. Ensure ClaimNb is non-negative integer.
    5. Cast categorical columns to ``category`` dtype.

    Returns
    -------
    pd.DataFrame — cleaned copy.
    """
    out = df.copy()

    # Remove invalid exposures
    out = out.loc[out["Exposure"] > 0].copy()

    # Cap exposure at 1
    out["Exposure"] = out["Exposure"].clip(upper=1.0)

    # IDpol as int
    out["IDpol"] = out["IDpol"].astype(int)

    # ClaimNb integrity
    out["ClaimNb"] = out["ClaimNb"].astype(int)
    out = out.loc[out["ClaimNb"] >= 0].copy()

    # Categoricals
    for col in ["VehBrand", "VehGas", "Area", "Region"]:
        if col in out.columns:
            out[col] = out[col].astype("category")

    out.reset_index(drop=True, inplace=True)
    return out


def clean_sev(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the severity dataset.

    Steps
    -----
    1. Drop rows with ClaimAmount <= 0.
    2. Cast IDpol to int.

    Returns
    -------
    pd.DataFrame — cleaned copy.
    """
    out = df.copy()
    out = out.loc[out["ClaimAmount"] > 0].copy()
    out["IDpol"] = out["IDpol"].astype(int)
    out.reset_index(drop=True, inplace=True)
    return out


def save_processed(
    freq: pd.DataFrame,
    sev: pd.DataFrame,
    freq_path: str | None = None,
    sev_path: str | None = None,
) -> tuple[str, str]:
    """Save cleaned DataFrames to CSV and return the output paths."""
    fp = freq_path or str(FREQ_PROCESSED)
    sp = sev_path or str(SEV_PROCESSED)
    freq.to_csv(fp, index=False)
    sev.to_csv(sp, index=False)
    return fp, sp
