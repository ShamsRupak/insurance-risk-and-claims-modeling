"""
Expected loss (pure premium) — combines frequency and severity predictions.

Pure Premium = E[Frequency] × E[Severity]
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_pure_premium(
    freq_pred: np.ndarray,
    sev_pred: np.ndarray | float,
) -> np.ndarray:
    """Compute per-policy pure premium.

    Parameters
    ----------
    freq_pred : array-like
        Predicted claim frequency (expected claim count per unit exposure).
    sev_pred : array-like or float
        Predicted average severity.  Can be a scalar (portfolio-level mean)
        or per-policy predictions from a severity model.

    Returns
    -------
    np.ndarray — pure premium for each policy.
    """
    return np.asarray(freq_pred) * np.asarray(sev_pred)


def portfolio_summary(
    df: pd.DataFrame,
    pure_premium_col: str = "PurePremium",
    exposure_col: str = "Exposure",
    claim_col: str = "ClaimNb",
) -> dict:
    """Compute aggregate portfolio-level metrics.

    Returns
    -------
    dict with keys:
        total_exposure, total_claims, observed_frequency,
        avg_pure_premium, total_expected_loss
    """
    total_exposure = df[exposure_col].sum()
    total_claims = df[claim_col].sum()
    observed_freq = total_claims / total_exposure if total_exposure > 0 else 0.0
    avg_pp = df[pure_premium_col].mean()
    total_el = (df[pure_premium_col] * df[exposure_col]).sum()

    return {
        "total_exposure": total_exposure,
        "total_claims": int(total_claims),
        "observed_frequency": observed_freq,
        "avg_pure_premium": avg_pp,
        "total_expected_loss": total_el,
    }


def risk_segment_summary(
    df: pd.DataFrame,
    segment_col: str,
    pure_premium_col: str = "PurePremium",
    exposure_col: str = "Exposure",
    claim_col: str = "ClaimNb",
) -> pd.DataFrame:
    """Group-by summary of pure premium by a segmentation variable."""
    grouped = df.groupby(segment_col).agg(
        n_policies=(exposure_col, "count"),
        total_exposure=(exposure_col, "sum"),
        total_claims=(claim_col, "sum"),
        avg_pure_premium=(pure_premium_col, "mean"),
    )
    grouped["observed_frequency"] = (
        grouped["total_claims"] / grouped["total_exposure"]
    )
    return grouped.sort_values("avg_pure_premium", ascending=False)
