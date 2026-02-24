"""
Evaluation metrics for frequency and severity models.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ===================================================================
# Deviance metrics  (unit deviance summed over observations)
# ===================================================================

def deviance_poisson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Poisson deviance.

    D = 2 * mean[ y*log(y/mu) - (y - mu) ]  where y=0 terms use the limit.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_pred = np.clip(y_pred, 1e-10, None)

    dev = np.where(
        y_true == 0,
        2 * y_pred,
        2 * (y_true * np.log(y_true / y_pred) - (y_true - y_pred)),
    )
    return float(dev.mean())


def deviance_gamma(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Gamma deviance (only valid for strictly positive y_true).

    D = 2 * mean[ -log(y/mu) + (y - mu)/mu ]
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_pred = np.clip(y_pred, 1e-10, None)

    dev = 2 * (-np.log(y_true / y_pred) + (y_true - y_pred) / y_pred)
    return float(dev.mean())


# ===================================================================
# Standard regression metrics
# ===================================================================

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(mean_absolute_error(y_true, y_pred))


# ===================================================================
# Lift / observed-vs-predicted helpers
# ===================================================================

def lift_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    exposure: np.ndarray | None = None,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Create a lift chart table (decile-based).

    Bins policies by predicted value and compares observed vs. predicted
    means within each bin.
    """
    tmp = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "exposure": exposure if exposure is not None else np.ones(len(y_true)),
    })
    tmp["bin"] = pd.qcut(tmp["y_pred"], q=n_bins, duplicates="drop")

    agg = tmp.groupby("bin", observed=True).agg(
        n=("y_true", "count"),
        total_exposure=("exposure", "sum"),
        observed_mean=("y_true", "mean"),
        predicted_mean=("y_pred", "mean"),
    )
    agg["lift"] = agg["observed_mean"] / agg["predicted_mean"]
    return agg.reset_index()


def gini_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Normalized Gini coefficient (2 * AUC - 1 style, via ranking)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    idx = np.argsort(y_pred)
    sorted_true = y_true[idx]
    cum = np.cumsum(sorted_true)
    gini_sum = cum.sum() / (sorted_true.sum() + 1e-10) - (n + 1) / 2
    return float(gini_sum / (n + 1e-10))
