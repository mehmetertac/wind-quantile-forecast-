"""Unit tests for quantile forecast evaluation metrics."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
from wind_quantile_forecast.config import QUANTILES
from wind_quantile_forecast.evaluation.metrics import (
    METRICS_TABLE_COLUMNS,
    PI_COVERAGE_TARGET,
    evaluate_quantile_forecast,
    log_fold_metrics,
    mae,
    mape,
    pi_coverage,
    pinball_loss,
    rmse,
    save_metrics_table,
)


def test_pinball_loss_perfect_prediction_zero() -> None:
    y = np.array([1.0, 2.0, 3.0])
    assert pinball_loss(y, y, quantile=0.5) == pytest.approx(0.0)


def test_pi_coverage_all_inside_interval() -> None:
    y = np.array([5.0, 10.0, 15.0])
    lo = np.array([0.0, 0.0, 0.0])
    hi = np.array([20.0, 20.0, 20.0])
    assert pi_coverage(y, lo, hi) == pytest.approx(1.0)


def test_pi_coverage_half_outside() -> None:
    y = np.array([5.0, 25.0])
    lo = np.array([0.0, 0.0])
    hi = np.array([10.0, 10.0])
    assert pi_coverage(y, lo, hi) == pytest.approx(0.5)


def test_mae_rmse_mape_on_p50() -> None:
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 27.0])
    assert mae(y_true, y_pred) == pytest.approx(7.0 / 3.0)
    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(17.0 / 3.0))
    assert mape(y_true, y_pred) == pytest.approx(40.0 / 3.0)


def test_mape_excludes_zero_actuals() -> None:
    y_true = np.array([0.0, 10.0])
    y_pred = np.array([1.0, 12.0])
    assert mape(y_true, y_pred) == pytest.approx(20.0)


def test_mape_all_zeros_returns_nan() -> None:
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([1.0, 2.0])
    assert np.isnan(mape(y_true, y_pred))


def test_evaluate_quantile_forecast_returns_expected_keys() -> None:
    y = np.array([10.0, 20.0, 30.0, 40.0])
    preds = {
        0.1: np.array([8.0, 18.0, 28.0, 38.0]),
        0.5: np.array([10.0, 20.0, 30.0, 40.0]),
        0.9: np.array([12.0, 22.0, 32.0, 42.0]),
    }
    metrics = evaluate_quantile_forecast(y, preds, quantiles=QUANTILES)
    assert metrics["n_test"] == 4.0
    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["rmse"] == pytest.approx(0.0)
    assert metrics["pinball_q50"] == pytest.approx(0.0)
    assert metrics["pi_coverage"] == pytest.approx(1.0)
    assert PI_COVERAGE_TARGET == pytest.approx(0.80)


def test_save_metrics_table_writes_csv(tmp_path) -> None:
    fold_metrics = pd.DataFrame(
        [
            {
                "fold": 1,
                "n_train": 80,
                "n_test": 20,
                "pinball_q10": 1.1,
                "pinball_q50": 0.5,
                "pinball_q90": 1.2,
                "pi_coverage": 0.75,
                "mae": 10.0,
                "rmse": 12.0,
                "mape": 2.5,
            },
            {
                "fold": 2,
                "n_train": 100,
                "n_test": 20,
                "pinball_q10": 1.0,
                "pinball_q50": 0.4,
                "pinball_q90": 1.0,
                "pi_coverage": 0.85,
                "mae": 9.0,
                "rmse": 11.0,
                "mape": 2.0,
            },
        ]
    )
    out = tmp_path / "metrics.csv"
    path = save_metrics_table(fold_metrics, out)
    assert path == out
    loaded = pd.read_csv(out)
    assert list(loaded.columns) == list(METRICS_TABLE_COLUMNS)
    assert len(loaded) == 3
    assert loaded.iloc[-1]["fold"] == 0
    assert loaded.iloc[-1]["pi_coverage"] == pytest.approx(0.80)


def test_log_fold_metrics_emits_info(caplog: pytest.LogCaptureFixture) -> None:
    metrics = {
        "n_test": 10.0,
        "pinball_q10": 1.2,
        "pinball_q50": 0.5,
        "pinball_q90": 1.1,
        "pi_coverage": 0.78,
        "mae": 3.0,
        "rmse": 4.0,
        "mape": 5.0,
    }
    test_logger = logging.getLogger("test.metrics")
    with caplog.at_level(logging.INFO, logger="test.metrics"):
        log_fold_metrics(2, metrics, log=test_logger)
    assert len(caplog.records) == 1
    msg = caplog.records[0].message
    assert "fold=2" in msg
    assert "pinball_q50=0.5000" in msg
    assert "pi_coverage=0.7800" in msg
    assert "mae=3.0000" in msg
