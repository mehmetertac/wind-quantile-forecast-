"""Unit tests for quantile GBM wrappers (LightGBM, XGBoost, CatBoost)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from wind_quantile_forecast.config import QUANTILES
from wind_quantile_forecast.evaluation.cv import evaluate_quantile_origin_cv
from wind_quantile_forecast.models.quantile_gbm import QuantileGBM, make_quantile_predict_fold

BACKENDS = ["lightgbm", "xgboost", "catboost"]
FAST_PARAMS = {"n_estimators": 20, "verbosity": -1}


def _synthetic_xy(n: int = 80) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    x1 = rng.uniform(0, 1, n)
    x2 = rng.uniform(0, 1, n)
    noise = rng.normal(0, 0.5, n)
    y = 10.0 + 5.0 * x1 - 3.0 * x2 + noise
    X = pd.DataFrame({"x1": x1, "x2": x2})
    return X, pd.Series(y, name="wind_mw")


@pytest.mark.parametrize("backend", BACKENDS)
def test_quantile_gbm_fit_predict_returns_all_quantiles(backend: str) -> None:
    X, y = _synthetic_xy()
    model = QuantileGBM(backend=backend, model_params=FAST_PARAMS)
    model.fit(X, y)
    preds = model.predict(X)
    assert set(preds) == set(QUANTILES)
    for q in QUANTILES:
        assert preds[q].shape == (len(X),)
        assert np.isfinite(preds[q]).all()


@pytest.mark.parametrize("backend", ["xgboost", "catboost"])
def test_quantile_gbm_per_quantile_mode(backend: str) -> None:
    X, y = _synthetic_xy()
    model = QuantileGBM(
        backend=backend,
        model_params=FAST_PARAMS,
        multi_quantile=False,
    )
    model.fit(X, y)
    preds = model.predict(X)
    assert set(preds) == set(QUANTILES)
    assert model._multi_model is None
    assert len(model._models) == len(QUANTILES)


def test_quantile_gbm_estimator_at_returns_fitted_model() -> None:
    X, y = _synthetic_xy()
    model = QuantileGBM(backend="lightgbm", model_params=FAST_PARAMS)
    model.fit(X, y)
    est = model.estimator_at(0.5)
    assert est is not None
    assert hasattr(est, "predict")


def test_quantile_gbm_estimator_at_unknown_quantile_raises() -> None:
    X, y = _synthetic_xy()
    model = QuantileGBM(backend="lightgbm", model_params=FAST_PARAMS)
    model.fit(X, y)
    with pytest.raises(KeyError, match="quantile"):
        model.estimator_at(0.75)


def test_quantile_gbm_estimator_at_multi_quantile_raises() -> None:
    X, y = _synthetic_xy()
    model = QuantileGBM(backend="catboost", model_params=FAST_PARAMS, multi_quantile=True)
    model.fit(X, y)
    with pytest.raises(ValueError, match="multi_quantile"):
        model.estimator_at(0.5)


def test_quantile_gbm_predict_before_fit_raises() -> None:
    X, _ = _synthetic_xy(n=10)
    model = QuantileGBM(backend="lightgbm")
    with pytest.raises(ValueError, match="fitted"):
        model.predict(X)


def test_quantile_gbm_unsupported_backend_raises() -> None:
    model = QuantileGBM(backend="lightgbm")
    model.backend = "unknown"  # type: ignore[assignment]
    with pytest.raises(NotImplementedError, match="not supported"):
        model._build_estimator(0.5)


@pytest.mark.parametrize("backend", BACKENDS)
def test_make_quantile_predict_fold_rolling_origin_cv(backend: str) -> None:
    n = 120
    idx = pd.date_range("2019-06-01", periods=n, freq="h", tz="Europe/Berlin")
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "valid_time": idx,
            "wind_mw": 1000.0 + np.arange(n) * 2.0 + rng.normal(0, 5, n),
            "hour": idx.hour,
            "feat_a": rng.uniform(0, 1, n),
        }
    )
    feature_cols = ["hour", "feat_a"]
    predict_fold = make_quantile_predict_fold(
        feature_cols,
        backend=backend,
        model_params={"n_estimators": 15, "verbosity": -1},
    )
    result = evaluate_quantile_origin_cv(
        df,
        feature_cols,
        predict_fold,
        n_splits=3,
        metrics_path=None,
    )
    assert len(result.fold_metrics) == 3
    assert result.oof_predictions is not None
    assert {"pred_q10", "pred_q50", "pred_q90"}.issubset(result.oof_predictions.columns)
    assert result.fold_metrics["pinball_q50"].notna().all()
    assert result.fold_metrics["n_test"].gt(0).all()
