"""
Claim severity models — Gamma GLM with log-link on positive claims.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

from src.utils.config import RANDOM_STATE, TEST_SIZE


class GammaSeverityModel:
    """Gamma GLM for modelling individual claim amounts (strictly positive).

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
        target: str = "ClaimAmount",
    ) -> "GammaSeverityModel":
        """Fit a Gamma GLM with log link."""
        X = sm.add_constant(df[self.feature_cols].values)
        y = df[target].values

        self.model_ = sm.GLM(
            y, X,
            family=sm.families.Gamma(link=sm.families.links.Log()),
        )
        self.results_ = self.model_.fit()
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict expected claim severity."""
        X = sm.add_constant(df[self.feature_cols].values)
        return self.results_.predict(X)

    def summary(self) -> str:
        """Return statsmodels summary table."""
        if self.results_ is None:
            raise RuntimeError("Model not fitted yet.")
        return self.results_.summary()


def train_test_severity(
    df: pd.DataFrame,
    feature_cols: list[str],
    target: str = "ClaimAmount",
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, GammaSeverityModel]:
    """Convenience: split data, fit Gamma GLM, return (train, test, model)."""
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    model = GammaSeverityModel(feature_cols)
    model.fit(train_df, target=target)
    return train_df, test_df, model
