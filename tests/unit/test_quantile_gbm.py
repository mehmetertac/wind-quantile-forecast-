"""Unit tests for LightGBM quantile GBM wrapper."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from wind_quantile_forecast.config import QUANTILES
from wind_quantile_forecast.evaluation.cv import evaluate_quantile_origin_cv
from wind_quantile_forecast.models.quantile_gbm import QuantileGBM, make_quantile_predict_fold


def _synthetic_xy(n: int = 80) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    x1 = rng.uniform(0, 1, n)
    x2 = rng.uniform(0, 1, n)
    noise = rng.normal(0, 0.5, n)
    y = 10.0 + 5.0 * x1 - 3.0 * x2 + noise
    X = pd.DataFrame({"x1": x1, "x2": x2})
    return X, pd.Series(y, name="wind_mw")


def test_quantile_gbm_fit_predict_returns_all_quantiles() -> None:
    X, y = _synthetic_xy()
    model = QuantileGBM(
        backend="lightgbm",
        model_params={"n_estimators": 20, "verbosity": -1},
    )
    model.fit(X, y)
    preds = model.predict(X)
    assert set(preds) == set(QUANTILES)
    for q in QUANTILES:
        assert preds[q].shape == (len(X),)
        assert np.isfinite(preds[q]).all()


def test_quantile_gbm_predict_before_fit_raises() -> None:
    X, _ = _synthetic_xy(n=10)
    model = QuantileGBM(backend="lightgbm")
    with pytest.raises(ValueError, match="fitted"):
        model.predict(X)


def test_quantile_gbm_unsupported_backend_raises() -> None:
    X, y = _synthetic_xy(n=10)
    model = QuantileGBM(backend="xgboost")
    with pytest.raises(NotImplementedError, match="xgboost"):
        model.fit(X, y)


def test_make_quantile_predict_fold_rolling_origin_cv() -> None:
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
