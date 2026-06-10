"""Unit tests for backend comparison table."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from wind_quantile_forecast.config import HOUR_SEASON_COL, QUANTILES
from wind_quantile_forecast.evaluation.comparison import (
    COMPARISON_TABLE_COLUMNS,
    BackendComparisonSpec,
    comparison_table_markdown,
    format_comparison_table,
    run_backend_comparison,
    summarize_backend_result,
    summarize_fold_metrics,
)
from wind_quantile_forecast.evaluation.cv import RollingOriginCVResult
from wind_quantile_forecast.features.target_encoding import add_hour_season_feature


def _synthetic_model_df(n: int = 120) -> tuple[pd.DataFrame, list[str]]:
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
    df = add_hour_season_feature(df)
    feature_cols = ["hour", HOUR_SEASON_COL, "feat_a"]
    return df, feature_cols


def test_summarize_fold_metrics_means_probabilistic_columns() -> None:
    fold_metrics = pd.DataFrame(
        {
            "pinball_q10": [0.2, 0.4],
            "pinball_q50": [1.0, 3.0],
            "pi_coverage": [0.7, 0.9],
            "mae": [10.0, 20.0],
        }
    )
    summary = summarize_fold_metrics(fold_metrics)
    assert summary["pinball_q10"] == pytest.approx(0.3)
    assert summary["pinball_q50"] == pytest.approx(2.0)
    assert summary["pi_coverage"] == pytest.approx(0.8)
    assert summary["mae"] == pytest.approx(15.0)


def test_format_comparison_table_has_stable_columns() -> None:
    rows = [
        {
            "backend": "lightgbm",
            "encoding": "target",
            "pinball_q50": 1.5,
            "train_time_sec": 3.2,
            "n_folds": 3,
        }
    ]
    table = format_comparison_table(rows)
    assert list(table.columns) == list(COMPARISON_TABLE_COLUMNS)


def test_summarize_backend_result_includes_train_time() -> None:
    fold_metrics = pd.DataFrame({"pinball_q50": [1.0, 2.0], "pi_coverage": [0.8, 0.8]})
    result = RollingOriginCVResult(fold_metrics=fold_metrics)
    row = summarize_backend_result(
        BackendComparisonSpec("xgboost", "target"),
        result,
        train_time_sec=12.5,
    )
    assert row["backend"] == "xgboost"
    assert row["encoding"] == "target"
    assert row["train_time_sec"] == pytest.approx(12.5)
    assert row["pinball_q50"] == pytest.approx(1.5)


def test_comparison_table_markdown_renders_header() -> None:
    table = format_comparison_table(
        [{"backend": "catboost", "encoding": "native", "train_time_sec": 1.0, "n_folds": 2}]
    )
    md = comparison_table_markdown(table)
    assert "| backend |" in md
    assert "catboost" in md


@pytest.mark.parametrize(
    "spec",
    [
        BackendComparisonSpec("lightgbm", "target"),
        BackendComparisonSpec("catboost", "native"),
    ],
)
def test_run_backend_comparison_single_spec(spec: BackendComparisonSpec) -> None:
    df, feature_cols = _synthetic_model_df()
    table = run_backend_comparison(
        df,
        feature_cols,
        specs=[spec],
        n_splits=3,
        model_params={"n_estimators": 10, "verbosity": -1},
        comparison_path=None,
    )
    assert len(table) == 1
    assert table["train_time_sec"].iloc[0] > 0
    for q in QUANTILES:
        col = f"pinball_q{int(q * 100):02d}"
        assert np.isfinite(table[col].iloc[0])
