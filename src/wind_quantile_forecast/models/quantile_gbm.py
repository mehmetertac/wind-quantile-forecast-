"""Gradient boosting quantile regression wrappers for LightGBM, XGBoost, CatBoost."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd

from wind_quantile_forecast.config import (
    DEFAULT_TARGET_ENCODING_SMOOTHING,
    QUANTILES,
    RANDOM_SEED,
    TARGET_COL,
)
from wind_quantile_forecast.features.target_encoding import (
    CatEncoding,
    apply_categorical_encoding,
    target_encoding_column,
)

Backend = Literal["lightgbm", "xgboost", "catboost"]

DEFAULT_LIGHTGBM_PARAMS: dict[str, Any] = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "num_leaves": 31,
    "verbosity": -1,
}

DEFAULT_XGBOOST_PARAMS: dict[str, Any] = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_depth": 6,
    "tree_method": "hist",
    "verbosity": 0,
}

DEFAULT_CATBOOST_PARAMS: dict[str, Any] = {
    "iterations": 100,
    "learning_rate": 0.1,
    "depth": 6,
    "verbose": False,
}


class QuantileGBM:
    """Train GBM quantile models using pinball loss."""

    def __init__(
        self,
        backend: Backend = "lightgbm",
        quantiles: list[float] | None = None,
        random_seed: int = RANDOM_SEED,
        model_params: dict[str, Any] | None = None,
        multi_quantile: bool = True,
        cat_features: Sequence[str] | None = None,
    ) -> None:
        """Configure a quantile GBM wrapper.

        Args:
            backend: ``lightgbm``, ``xgboost``, or ``catboost``.
            quantiles: Target quantile levels (default P10/P50/P90).
            random_seed: Random seed for reproducible fits.
            model_params: Extra estimator kwargs (e.g. ``n_estimators``).
            multi_quantile: When ``True`` and the backend supports it, fit one
                joint model for all quantiles (XGBoost ``reg:quantileerror`` with
                a list ``quantile_alpha``; CatBoost ``MultiQuantile``). LightGBM
                always trains one model per quantile.
            cat_features: Column names passed to CatBoost as categorical (native mode).
        """
        self.backend = backend
        self.quantiles = list(quantiles or QUANTILES)
        self.random_seed = random_seed
        self.model_params = dict(model_params or {})
        self.multi_quantile = multi_quantile
        self.cat_features = list(cat_features or [])
        self._models: dict[float, object] = {}
        self._multi_model: object | None = None
        self._feature_names: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> QuantileGBM:
        """Fit quantile models on ``(X, y)``.

        LightGBM uses ``objective="quantile"`` with ``alpha`` per quantile.
        XGBoost uses ``objective="reg:quantileerror"`` (single or multi-quantile).
        CatBoost uses ``Quantile:alpha=...`` or ``MultiQuantile``.

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
        self._multi_model = None

        if self._use_multi_quantile():
            estimator = self._build_multi_estimator()
            self._fit_estimator(estimator, X, y_arr)
            self._multi_model = estimator
            return self

        for q in self.quantiles:
            estimator = self._build_estimator(q)
            self._fit_estimator(estimator, X, y_arr)
            self._models[q] = estimator
        return self

    def _fit_estimator(self, estimator: object, X: pd.DataFrame, y: np.ndarray) -> None:
        if self.backend == "catboost" and self.cat_features:
            estimator.fit(X, y, cat_features=list(self.cat_features))
            return
        estimator.fit(X, y)

    def predict(self, X: pd.DataFrame) -> dict[float, np.ndarray]:
        """Predict all quantiles for new observations.

        Args:
            X: Feature matrix with the same columns used at fit time.

        Returns:
            Dict mapping quantile -> predicted array (e.g. ``{0.1: ..., 0.5: ..., 0.9: ...}``).
        """
        if self._multi_model is None and not self._models:
            msg = "QuantileGBM must be fitted before predict"
            raise ValueError(msg)

        if self._feature_names is not None and isinstance(X, pd.DataFrame):
            missing = set(self._feature_names) - set(X.columns)
            if missing:
                msg = f"predict input missing feature columns: {sorted(missing)}"
                raise ValueError(msg)
            X = X[self._feature_names]

        if self._multi_model is not None:
            raw = np.asarray(self._multi_model.predict(X), dtype=float)
            if raw.ndim == 1:
                return {self.quantiles[0]: raw}
            return {q: raw[:, i] for i, q in enumerate(self.quantiles)}

        return {
            q: np.asarray(self._models[q].predict(X), dtype=float) for q in self.quantiles
        }

    def _use_multi_quantile(self) -> bool:
        return (
            self.multi_quantile
            and self.backend in ("xgboost", "catboost")
            and len(self.quantiles) > 1
        )

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
        if self.backend == "xgboost":
            import xgboost as xgb

            params = self._xgboost_params()
            return xgb.XGBRegressor(
                objective="reg:quantileerror",
                quantile_alpha=quantile,
                random_state=self.random_seed,
                **params,
            )
        if self.backend == "catboost":
            from catboost import CatBoostRegressor

            params = self._catboost_params()
            return CatBoostRegressor(
                loss_function=f"Quantile:alpha={quantile}",
                random_seed=self.random_seed,
                **params,
            )
        raise NotImplementedError(
            f"QuantileGBM backend {self.backend!r} not supported; "
            "use 'lightgbm', 'xgboost', or 'catboost'"
        )

    def _build_multi_estimator(self):
        if self.backend == "xgboost":
            import xgboost as xgb

            params = self._xgboost_params()
            return xgb.XGBRegressor(
                objective="reg:quantileerror",
                quantile_alpha=list(self.quantiles),
                random_state=self.random_seed,
                **params,
            )
        if self.backend == "catboost":
            from catboost import CatBoostRegressor

            alpha_str = ",".join(str(q) for q in self.quantiles)
            params = self._catboost_params()
            return CatBoostRegressor(
                loss_function=f"MultiQuantile:alpha={alpha_str}",
                random_seed=self.random_seed,
                **params,
            )
        msg = f"multi-quantile training is not supported for backend {self.backend!r}"
        raise NotImplementedError(msg)

    def _xgboost_params(self) -> dict[str, Any]:
        params = {**DEFAULT_XGBOOST_PARAMS, **self.model_params}
        verbosity = params.get("verbosity")
        if verbosity is not None and verbosity < 0:
            params["verbosity"] = 0
        return params

    def _catboost_params(self) -> dict[str, Any]:
        user = dict(self.model_params)
        if "n_estimators" in user:
            user["iterations"] = user.pop("n_estimators")
        user.pop("verbosity", None)
        params = {**DEFAULT_CATBOOST_PARAMS, **user}
        params.pop("n_estimators", None)
        return params


def _resolve_model_columns(
    feature_cols: Sequence[str],
    cat_col: str | None,
    encoding: CatEncoding,
) -> tuple[list[str], list[str] | None]:
    """Map modeling columns and CatBoost categorical feature list per encoding mode."""
    cols = list(feature_cols)
    if cat_col is None or encoding == "none" or cat_col not in cols:
        return cols, None
    if encoding == "target":
        encoded = target_encoding_column(cat_col)
        cols = [encoded if c == cat_col else c for c in cols]
        return cols, None
    if encoding == "native":
        return cols, [cat_col]
    msg = f"unsupported encoding mode: {encoding!r}"
    raise ValueError(msg)


def make_quantile_predict_fold(
    feature_cols: Sequence[str],
    *,
    target_col: str = TARGET_COL,
    backend: Backend = "lightgbm",
    quantiles: Sequence[float] | None = None,
    random_seed: int = RANDOM_SEED,
    model_params: dict[str, Any] | None = None,
    multi_quantile: bool = True,
    cat_col: str | None = None,
    cat_encoding: CatEncoding = "none",
    encoding_smoothing: float = DEFAULT_TARGET_ENCODING_SMOOTHING,
) -> Callable[[pd.DataFrame, pd.DataFrame], dict[float, np.ndarray]]:
    """Build a ``predict_fold`` callback for :func:`evaluate_quantile_origin_cv`.

    Each fold fits a fresh :class:`QuantileGBM` on ``train_df`` and predicts
    P10/P50/P90 (or custom quantiles) on ``test_df``.

    Args:
        feature_cols: Modeling feature column names present in both frames.
        target_col: Target column (default ``wind_mw``).
        backend: GBM backend (``lightgbm``, ``xgboost``, or ``catboost``).
        quantiles: Quantile levels; defaults to ``config.QUANTILES``.
        random_seed: Random seed passed to each fold's models.
        model_params: Extra estimator kwargs (e.g. ``n_estimators``).
        multi_quantile: Use joint multi-quantile training for XGBoost/CatBoost.
        cat_col: High-cardinality categorical column (e.g. ``hour_season``).
        cat_encoding: ``"target"`` encodes inside each fold; ``"native"`` keeps
            the raw column for CatBoost; ``"none"`` skips encoding.
        encoding_smoothing: Smoothing factor for fold-wise target encoding.

    Returns:
        ``(train_df, test_df) -> {quantile: np.ndarray}`` suitable for rolling-origin CV.
    """
    if cat_encoding == "native" and backend != "catboost":
        msg = "native categorical encoding is only supported for catboost backend"
        raise ValueError(msg)

    model_cols, native_cat = _resolve_model_columns(feature_cols, cat_col, cat_encoding)
    q_list = list(quantiles or QUANTILES)

    def predict_fold(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> dict[float, np.ndarray]:
        tr = train_df
        te = test_df
        cols = list(model_cols)
        cat_features = native_cat

        if cat_col is not None and cat_encoding != "none":
            tr, te, encoded_cols = apply_categorical_encoding(
                train_df,
                test_df,
                cat_col,
                target_col,
                encoding=cat_encoding,
                smoothing=encoding_smoothing,
            )
            if cat_encoding == "target" and encoded_cols:
                cols = [encoded_cols[0] if c == cat_col else c for c in model_cols]

        model = QuantileGBM(
            backend=backend,
            quantiles=q_list,
            random_seed=random_seed,
            model_params=model_params,
            multi_quantile=multi_quantile,
            cat_features=cat_features,
        )
        model.fit(tr[cols], tr[target_col])
        return model.predict(te[cols])

    return predict_fold
