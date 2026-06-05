"""Gradient boosting quantile regression wrappers for LightGBM, XGBoost, CatBoost."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from wind_quantile_forecast.config import QUANTILES

if TYPE_CHECKING:
    import pandas as pd

Backend = Literal["lightgbm", "xgboost", "catboost"]


class QuantileGBM:
    """Train separate GBM models per quantile using pinball loss."""

    def __init__(
        self,
        backend: Backend = "lightgbm",
        quantiles: list[float] | None = None,
        random_seed: int = 42,
    ) -> None:
        self.backend = backend
        self.quantiles = quantiles or QUANTILES
        self.random_seed = random_seed
        self._models: dict[float, object] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> QuantileGBM:
        """Fit one GBM per quantile.

        Args:
            X: Feature matrix.
            y: Target wind generation values.

        Returns:
            Fitted self.

        Raises:
            NotImplementedError: Training logic not yet implemented.
        """
        raise NotImplementedError(f"QuantileGBM.fit not yet implemented for backend={self.backend}")

    def predict(self, X: pd.DataFrame) -> dict[float, np.ndarray]:
        """Predict all quantiles for new observations.

        Args:
            X: Feature matrix.

        Returns:
            Dict mapping quantile -> predicted array.

        Raises:
            NotImplementedError: Prediction logic not yet implemented.
        """
        raise NotImplementedError(
            f"QuantileGBM.predict not yet implemented for backend={self.backend}"
        )
