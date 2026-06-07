"""Gradient boosting quantile regression wrappers for LightGBM, XGBoost, CatBoost."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd

from wind_quantile_forecast.config import QUANTILES, RANDOM_SEED, TARGET_COL

Backend = Literal["lightgbm", "xgboost", "catboost"]

DEFAULT_LIGHTGBM_PARAMS: dict[str, Any] = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "num_leaves": 31,
    "verbosity": -1,
}


class QuantileGBM:
    """Train separate GBM models per quantile using pinball loss."""

    def __init__(
        self,
        backend: Backend = "lightgbm",
        quantiles: list[float] | None = None,
        random_seed: int = RANDOM_SEED,
        model_params: dict[str, Any] | None = None,
    ) -> None:
        self.backend = backend
        self.quantiles = list(quantiles or QUANTILES)
        self.random_seed = random_seed
        self.model_params = dict(model_params or {})
        self._models: dict[float, object] = {}
        self._feature_names: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> QuantileGBM:
        """Fit one GBM per quantile.

        LightGBM uses ``objective="quantile"`` with ``alpha`` set to each target
        quantile (0.1 / 0.5 / 0.9 for P10 / P50 / P90).

        Args:
            X: Feature matrix.
            y: Target wind generation values.

        Returns:
            Fitted self.
        """
        if isinstance(X, pd.DataFrame):
            self._feature_names = list(X.columns)
        else:
            self._feature_names = None

        y_arr = np.asarray(y, dtype=float)
        self._models = {}
        for q in self.quantiles:
            estimator = self._build_estimator(q)
            estimator.fit(X, y_arr)
            self._models[q] = estimator
        return self

    def predict(self, X: pd.DataFrame) -> dict[float, np.ndarray]:
        """Predict all quantiles for new observations.

        Args:
            X: Feature matrix with the same columns used at fit time.

        Returns:
            Dict mapping quantile -> predicted array (e.g. ``{0.1: ..., 0.5: ..., 0.9: ...}``).
        """
        if not self._models:
            msg = "QuantileGBM must be fitted before predict"
            raise ValueError(msg)

        if self._feature_names is not None and isinstance(X, pd.DataFrame):
            missing = set(self._feature_names) - set(X.columns)
            if missing:
                msg = f"predict input missing feature columns: {sorted(missing)}"
                raise ValueError(msg)
            X = X[self._feature_names]

        return {
            q: np.asarray(self._models[q].predict(X), dtype=float) for q in self.quantiles
        }

    def _build_estimator(self, quantile: float):
        if self.backend == "lightgbm":
            import lightgbm as lgb

            params = {**DEFAULT_LIGHTGBM_PARAMS, **self.model_params}
            return lgb.LGBMRegressor(
                objective="quantile",
                alpha=quantile,
                random_state=self.random_seed,
                **params,
            )
        raise NotImplementedError(
            f"QuantileGBM backend {self.backend!r} not yet implemented; use 'lightgbm'"
        )


def make_quantile_predict_fold(
    feature_cols: Sequence[str],
    *,
    target_col: str = TARGET_COL,
    backend: Backend = "lightgbm",
    quantiles: Sequence[float] | None = None,
    random_seed: int = RANDOM_SEED,
    model_params: dict[str, Any] | None = None,
) -> Callable[[pd.DataFrame, pd.DataFrame], dict[float, np.ndarray]]:
    """Build a ``predict_fold`` callback for :func:`evaluate_quantile_origin_cv`.

    Each fold fits a fresh :class:`QuantileGBM` on ``train_df`` and predicts
    P10/P50/P90 (or custom quantiles) on ``test_df``.

    Args:
        feature_cols: Modeling feature column names present in both frames.
        target_col: Target column (default ``wind_mw``).
        backend: GBM backend (only ``lightgbm`` implemented).
        quantiles: Quantile levels; defaults to ``config.QUANTILES``.
        random_seed: Random seed passed to each fold's models.
        model_params: Extra estimator kwargs (e.g. ``n_estimators``).

    Returns:
        ``(train_df, test_df) -> {quantile: np.ndarray}`` suitable for rolling-origin CV.
    """
    cols = list(feature_cols)
    q_list = list(quantiles or QUANTILES)

    def predict_fold(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> dict[float, np.ndarray]:
        model = QuantileGBM(
            backend=backend,
            quantiles=q_list,
            random_seed=random_seed,
            model_params=model_params,
        )
        model.fit(train_df[cols], train_df[target_col])
        return model.predict(test_df[cols])

    return predict_fold
