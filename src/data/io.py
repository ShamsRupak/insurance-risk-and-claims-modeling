"""
Data I/O — load and validate the French MTPL2 frequency and severity CSVs.
"""
from __future__ import annotations

import pandas as pd

from src.utils.config import (
    FREQ_PATH,
    SEV_PATH,
    FREQ_REQUIRED_COLS,
    SEV_REQUIRED_COLS,
)


def load_freq(path: str | None = None) -> pd.DataFrame:
    """Load the frequency (policy-level) dataset.

    Parameters
    ----------
    path : str or None
        Override path; defaults to ``config.FREQ_PATH``.

    Returns
    -------
    pd.DataFrame
    """
    p = path or str(FREQ_PATH)
    df = pd.read_csv(p, low_memory=False)
    _validate_schema(df, FREQ_REQUIRED_COLS, "frequency")
    return df


def load_sev(path: str | None = None) -> pd.DataFrame:
    """Load the severity (claim-level) dataset.

    Parameters
    ----------
    path : str or None
        Override path; defaults to ``config.SEV_PATH``.

    Returns
    -------
    pd.DataFrame
    """
    p = path or str(SEV_PATH)
    df = pd.read_csv(p, low_memory=False)
    _validate_schema(df, SEV_REQUIRED_COLS, "severity")
    return df


def _validate_schema(
    df: pd.DataFrame, required: list[str], label: str
) -> None:
    """Raise ``ValueError`` if *required* columns are missing from *df*."""
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(
            f"{label} dataset missing required columns: {missing}"
        )
