"""Unit tests for reliability / calibration metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from wind_quantile_forecast.config import QUANTILES
from wind_quantile_forecast.evaluation.reliability import (
    build_calibration_table,
    compute_reliability_curve,
    marginal_quantile_coverage,
    quantile_calibration_gap,
    reliability_from_oof,
)


def test_marginal_quantile_coverage_perfect_calibration() -> None:
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 5000)
    predictions = {
        0.1: np.quantile(y, 0.1) * np.ones_like(y),
        0.5: np.quantile(y, 0.5) * np.ones_like(y),
        0.9: np.quantile(y, 0.9) * np.ones_like(y),
    }
    coverage = marginal_quantile_coverage(y, predictions)
    assert coverage[0.1] == pytest.approx(0.1, abs=0.02)
    assert coverage[0.5] == pytest.approx(0.5, abs=0.02)
    assert coverage[0.9] == pytest.approx(0.9, abs=0.02)


def test_compute_reliability_curve_reports_marginal_coverage() -> None:
    y = np.arange(1.0, 11.0)
    predictions = {
        0.1: np.full(10, 1.5),
        0.5: np.full(10, 5.5),
        0.9: np.full(10, 9.5),
    }
    nominal, observed = compute_reliability_curve(y, predictions)
    assert nominal.tolist() == [0.1, 0.5, 0.9]
    assert observed[0] == pytest.approx(0.1)
    assert observed[1] == pytest.approx(0.5)
    assert observed[2] == pytest.approx(0.9)
    assert quantile_calibration_gap(observed, nominal) == pytest.approx(0.0)


def test_compute_reliability_curve_detects_miscalibration() -> None:
    y = np.arange(1.0, 11.0)
    predictions = {
        0.1: np.full(10, 3.0),
        0.5: np.full(10, 6.0),
        0.9: np.full(10, 8.0),
    }
    nominal, observed = compute_reliability_curve(y, predictions)
    gap = quantile_calibration_gap(observed, nominal)
    assert gap > 0


def test_reliability_from_oof_builds_curve() -> None:
    oof = pd.DataFrame(
        {
            "wind_mw": [10.0, 20.0, 30.0, 40.0],
            "pred_q10": [12.0, 22.0, 32.0, 42.0],
            "pred_q50": [15.0, 25.0, 28.0, 38.0],
            "pred_q90": [18.0, 28.0, 35.0, 45.0],
        }
    )
    nominal, observed = reliability_from_oof(oof)
    assert len(nominal) == 3
    assert observed[1] == pytest.approx(0.5)


def test_build_calibration_table_includes_interval_row() -> None:
    oof = pd.DataFrame(
        {
            "wind_mw": [10.0, 20.0, 30.0, 40.0],
            "pred_q10": [8.0, 18.0, 28.0, 38.0],
            "pred_q50": [10.0, 20.0, 30.0, 40.0],
            "pred_q90": [12.0, 22.0, 32.0, 42.0],
        }
    )
    table = build_calibration_table(oof, quantiles=QUANTILES)
    assert "pi_interval" in table["metric"].values
    assert table.loc[table["metric"] == "pi_interval", "observed"].iloc[0] == pytest.approx(1.0)
