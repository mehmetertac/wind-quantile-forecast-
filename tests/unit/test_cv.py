"""Tests for rolling-origin cross-validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import TimeSeriesSplit
from wind_quantile_forecast.evaluation.cv import (
    RollingOriginCVResult,
    RollingOriginSplit,
    rolling_origin_time_folds,
    run_rolling_origin_cv,
)


def _hourly_frame(n_hours: int = 120) -> pd.DataFrame:
    idx = pd.date_range("2019-06-01", periods=n_hours, freq="h", tz="Europe/Berlin")
    return pd.DataFrame(
        {
            "valid_time": idx,
            "wind_mw": np.linspace(1000, 2000, n_hours),
            "hour": idx.hour,
        }
    )


def test_rolling_origin_time_folds_expanding() -> None:
    times = pd.date_range("2020-01-01", periods=30, freq="h", tz="UTC")
    folds = rolling_origin_time_folds(times, n_splits=4)
    assert len(folds) == 4
    f0, f1 = folds[0], folds[1]
    assert f0.train_times.max() < f0.test_times.min()
    assert len(f1.train_times) > len(f0.train_times)


def test_rolling_origin_time_folds_rolling_mode() -> None:
    times = pd.date_range("2020-01-01", periods=40, freq="h", tz="UTC")
    folds = rolling_origin_time_folds(times, n_splits=4, mode="rolling", max_train_size=8)
    for fold in folds:
        assert len(fold.train_times) <= 8


def test_rolling_origin_time_folds_gap() -> None:
    times = pd.date_range("2020-01-01", periods=30, freq="h", tz="UTC")
    folds = rolling_origin_time_folds(times, n_splits=3, gap=2)
    fold = folds[0]
    assert fold.train_times.max() < fold.test_times.min() - pd.Timedelta(hours=1)


def test_rolling_origin_time_folds_too_few_times_raises() -> None:
    times = pd.date_range("2020-01-01", periods=3, freq="h", tz="UTC")
    with pytest.raises(ValueError, match="unique timestamps"):
        rolling_origin_time_folds(times, n_splits=5)


def test_rolling_origin_split_sklearn_backend_matches_tscv() -> None:
    n = 50
    splitter = RollingOriginSplit(n_splits=4, backend="sklearn")
    custom = list(splitter.split(np.zeros((n, 1))))

    tscv = TimeSeriesSplit(n_splits=4)
    expected = list(tscv.split(np.zeros((n, 1))))
    assert len(custom) == len(expected)
    for (tr, te), (etr, ete) in zip(custom, expected, strict=True):
        np.testing.assert_array_equal(tr, etr)
        np.testing.assert_array_equal(te, ete)


def test_rolling_origin_split_custom_yields_monotonic_folds() -> None:
    df = _hourly_frame()
    splitter = RollingOriginSplit(n_splits=5)
    splits = list(splitter.split(df, groups=df["valid_time"]))
    assert len(splits) == 5
    train_idx, test_idx = splits[0]
    assert train_idx.max() < test_idx.min()


def test_split_frame_yields_disjoint_train_test() -> None:
    df = _hourly_frame()
    splitter = RollingOriginSplit(n_splits=4)
    for train, test, fold in splitter.split_frame(df):
        assert train["valid_time"].max() < test["valid_time"].min()
        assert fold.n_train == train["valid_time"].nunique()
        assert fold.n_test == test["valid_time"].nunique()


def test_run_rolling_origin_cv_harness() -> None:
    df = _hourly_frame()

    def evaluate_fold(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        fold,
    ) -> dict[str, float]:
        baseline = float(train_df["wind_mw"].mean())
        preds = np.full(len(test_df), baseline)
        y = test_df["wind_mw"].to_numpy()
        return {"mae": float(np.mean(np.abs(y - preds))), "fold_id": float(fold.fold)}

    result = run_rolling_origin_cv(df, evaluate_fold, n_splits=4, log_folds=False)
    assert isinstance(result, RollingOriginCVResult)
    assert len(result.fold_metrics) == 4
    assert "mae" in result.fold_metrics.columns
    assert result.fold_metrics["n_test"].gt(0).all()
