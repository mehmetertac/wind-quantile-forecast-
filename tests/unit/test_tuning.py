"""Unit tests for hyperparameter tuning."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from wind_quantile_forecast.evaluation.tuning import (
    default_final_config,
    mean_pinball_cv_score,
    save_final_model_params,
    tune_lightgbm_quantile_cv,
)


def test_mean_pinball_cv_score_excludes_summary_row() -> None:
    fold_metrics = pd.DataFrame(
        {
            "fold": [1, 2, 0],
            "pinball_q10": [1.0, 3.0, 2.0],
            "pinball_q50": [2.0, 4.0, 3.0],
            "pinball_q90": [3.0, 5.0, 4.0],
        }
    )
    score = mean_pinball_cv_score(fold_metrics)
    assert score == pytest.approx((1 + 3 + 2 + 4 + 3 + 5) / 6)


def test_mean_pinball_cv_score_empty_raises() -> None:
    with pytest.raises(ValueError, match="no fold metrics"):
        mean_pinball_cv_score(pd.DataFrame())


def test_default_final_config_includes_backend_metadata() -> None:
    config = default_final_config({"n_estimators": 120, "verbosity": -1})
    assert config["backend"] == "lightgbm"
    assert config["enforce_monotonic"] is True
    assert config["model_params"]["n_estimators"] == 120


def test_tune_lightgbm_quantile_cv_runs_small_study() -> None:
    n = 80
    idx = pd.date_range("2019-06-01", periods=n, freq="h", tz="Europe/Berlin")
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "valid_time": idx,
            "wind_mw": 1000.0 + np.arange(n) * 2.0 + rng.normal(0, 5, n),
            "hour": idx.hour,
            "hour_season": (idx.hour.astype(str) + "_1"),
            "feat_a": rng.uniform(0, 1, n),
        }
    )
    feature_cols = ["hour", "hour_season", "feat_a"]
    result = tune_lightgbm_quantile_cv(
        df,
        feature_cols,
        n_trials=2,
        n_splits=2,
    )
    assert result.best_score > 0
    assert "n_estimators" in result.best_params
    assert len(result.trials) == 2


def test_save_final_model_params_writes_json(tmp_path) -> None:
    path = tmp_path / "final.json"
    save_final_model_params({"backend": "lightgbm"}, path=path, cv_summary={"mean_pinball": 1.5})
    text = path.read_text(encoding="utf-8")
    assert "lightgbm" in text
    assert "mean_pinball" in text
