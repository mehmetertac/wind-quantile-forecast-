"""Unit tests for SHAP explanation plots."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from wind_quantile_forecast.interpret.shap_explain import explain_model
from wind_quantile_forecast.models.quantile_gbm import QuantileGBM

FAST_PARAMS = {"n_estimators": 15, "verbosity": -1}


def _synthetic_xy(n: int = 60) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    x1 = rng.uniform(0, 10, n)
    x2 = rng.uniform(0, 5, n)
    noise = rng.normal(0, 0.5, n)
    y = 100.0 + 8.0 * x1 - 2.0 * x2 + noise
    X = pd.DataFrame(
        {
            "nwp_wind_speed_hub_mps": x1,
            "wind_mw_lag_24": x2,
            "wind_mw_roll24_mean": x1 * 0.5,
            "hour_season_te": rng.uniform(800, 1200, n),
            "extra": rng.normal(0, 1, n),
        }
    )
    return X, pd.Series(y, name="wind_mw")


def test_explain_model_writes_summary_and_dependence_plots(tmp_path: Path) -> None:
    X, y = _synthetic_xy()
    model = QuantileGBM(
        backend="lightgbm",
        quantiles=[0.5, 0.9],
        model_params=FAST_PARAMS,
    )
    model.fit(X, y)

    out = explain_model(model, X, quantile=0.5, output_dir=tmp_path, max_samples=50)
    summary = out / "shap_summary_p50.png"
    dep_wind = out / "shap_dependence_p50_nwp_wind_speed_hub_mps.png"

    assert summary.is_file() and summary.stat().st_size > 0
    assert dep_wind.is_file() and dep_wind.stat().st_size > 0


def test_explain_model_missing_dependence_features_raises(tmp_path: Path) -> None:
    X, y = _synthetic_xy()
    X = X[["extra"]]
    model = QuantileGBM(backend="lightgbm", quantiles=[0.5], model_params=FAST_PARAMS)
    model.fit(X, y)

    with pytest.raises(ValueError, match="no dependence features"):
        explain_model(
            model,
            X,
            quantile=0.5,
            dependence_features=["nwp_wind_speed_hub_mps"],
            output_dir=tmp_path,
        )


def test_explain_model_skips_missing_optional_dependence(tmp_path: Path) -> None:
    X, y = _synthetic_xy()
    model = QuantileGBM(backend="lightgbm", quantiles=[0.5], model_params=FAST_PARAMS)
    model.fit(X, y)

    explain_model(
        model,
        X,
        quantile=0.5,
        dependence_features=["nwp_wind_speed_hub_mps", "not_a_column"],
        output_dir=tmp_path,
        max_samples=50,
    )
    assert (tmp_path / "shap_summary_p50.png").is_file()
    assert (tmp_path / "shap_dependence_p50_nwp_wind_speed_hub_mps.png").is_file()
