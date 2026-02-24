"""
Claim frequency models — Poisson GLM with exposure offset.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

from src.utils.config import RANDOM_STATE, TEST_SIZE


class PoissonFrequencyModel:
    """Poisson GLM for modelling claim counts with an exposure offset.

    Parameters
    ----------
    feature_cols : list[str]
        Columns used as predictors (should already be numeric / encoded).
    """

    def __init__(self, feature_cols: list[str]):
        self.feature_cols = feature_cols
        self.model_ = None
        self.results_ = None

    def fit(
        self,
        df: pd.DataFrame,
        target: str = "ClaimNb",
        exposure: str = "Exposure",
    ) -> "PoissonFrequencyModel":
        """Fit a Poisson GLM with ``log(exposure)`` as offset."""
        X = sm.add_constant(df[self.feature_cols].values)
        y = df[target].values
        offset = np.log(df[exposure].values)

        self.model_ = sm.GLM(
            y, X, family=sm.families.Poisson(), offset=offset
        )
        self.results_ = self.model_.fit()
        return self

    def predict(self, df: pd.DataFrame, exposure: str = "Exposure") -> np.ndarray:
        """Predict expected claim count (lambda * exposure)."""
        X = sm.add_constant(df[self.feature_cols].values)
        offset = np.log(df[exposure].values)
        return self.results_.predict(X, offset=offset)

    def summary(self) -> str:
        """Return statsmodels summary table."""
        if self.results_ is None:
            raise RuntimeError("Model not fitted yet.")
        return self.results_.summary()


def train_test_frequency(
    df: pd.DataFrame,
    feature_cols: list[str],
    target: str = "ClaimNb",
    exposure: str = "Exposure",
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, PoissonFrequencyModel]:
    """Convenience: split data, fit Poisson GLM, return (train, test, model)."""
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    model = PoissonFrequencyModel(feature_cols)
    model.fit(train_df, target=target, exposure=exposure)
    return train_df, test_df, model
