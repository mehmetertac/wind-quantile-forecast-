"""Tests for feature matrix assembly and day-ahead leakage guards."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from wind_quantile_forecast.config import GENERATION_COL, TARGET_COL
from wind_quantile_forecast.data.dataset import prepare_day_ahead_dataset
from wind_quantile_forecast.features.lags import add_lag_features
from wind_quantile_forecast.features.matrix import (
    assemble_feature_matrix,
    validate_no_target_leakage,
)
from wind_quantile_forecast.features.weather import (
    AIR_DENSITY_COL,
    WIND_DIRECTION_COS_COL,
    WIND_DIRECTION_SIN_COL,
    add_weather_driver_features,
)


def _sample_features(n_hours: int = 72, leads: list[int] | None = None) -> pd.DataFrame:
    leads = leads or [24]
    idx = pd.date_range("2019-06-01", periods=n_hours, freq="h", tz="Europe/Berlin")
    rows: list[dict[str, object]] = []
    for vt in idx:
        for lead in leads:
            init = vt - pd.Timedelta(hours=lead)
            rows.append(
                {
                    "valid_time": vt,
                    "init_time": init,
                    "lead_hours": lead,
                    GENERATION_COL: 1000.0 + vt.hour * 10 + vt.day,
                    "hour": vt.hour,
                    "day_of_week": vt.dayofweek,
                    "month": vt.month,
                    "week_of_year": int(vt.isocalendar().week),
                    "is_weekend": int(vt.dayofweek >= 5),
                    "is_holiday": 0,
                    "days_since_last_holiday": 3.0,
                    "days_to_next_holiday": 10.0,
                    "sin_hour": np.sin(2 * np.pi * vt.hour / 24),
                    "cos_hour": np.cos(2 * np.pi * vt.hour / 24),
                    "sin_day_of_year": 0.1,
                    "cos_day_of_year": 0.9,
                    "nwp_wind_speed_10m": 5.0 + vt.hour * 0.1,
                    "nwp_wind_speed_hub_mps": 6.0 + vt.hour * 0.1,
                    "nwp_wind_direction_10m": 180.0 + vt.hour,
                    "nwp_t2m_k": 288.0,
                    "era5_wind_speed_hub_mps": 99.0,
                }
            )
    return pd.DataFrame(rows)


def _gen_history(n_hours: int = 96) -> pd.Series:
    idx = pd.date_range("2019-06-01", periods=n_hours, freq="h", tz="Europe/Berlin")
    return pd.Series(500.0 + np.arange(n_hours) * 3.0, index=idx, name=GENERATION_COL)


def test_weather_driver_features_adds_sin_cos_and_density() -> None:
    df = _sample_features(n_hours=2)
    out = add_weather_driver_features(df)
    assert WIND_DIRECTION_SIN_COL in out.columns
    assert WIND_DIRECTION_COS_COL in out.columns
    assert AIR_DENSITY_COL in out.columns
    assert out[AIR_DENSITY_COL].between(1.0, 1.4).all()


def test_lag_features_join_at_init_time_not_valid_time() -> None:
    """Lag values must reflect init_time, not leak post-issue target at valid_time."""
    idx = pd.date_range("2019-06-03", periods=30, freq="h", tz="Europe/Berlin")
    gen = pd.Series(np.arange(30, dtype=float) * 100.0, index=idx, name=GENERATION_COL)

    vt = idx[-1]
    init = vt - pd.Timedelta(hours=24)
    row = pd.DataFrame(
        [
            {
                "valid_time": vt,
                "init_time": init,
                "lead_hours": 24,
                TARGET_COL: 9999.0,
            }
        ]
    )
    out = add_lag_features(row, gen, lags=[1], windows=[])

    expected_lag1 = float(gen.loc[init - pd.Timedelta(hours=1)])
    assert out[f"{TARGET_COL}_lag_1"].iloc[0] == pytest.approx(expected_lag1)
    leaked = float(gen.loc[vt - pd.Timedelta(hours=1)])
    assert out[f"{TARGET_COL}_lag_1"].iloc[0] != pytest.approx(leaked)


def test_assemble_feature_matrix_columns() -> None:
    features = _sample_features()
    day_ahead = prepare_day_ahead_dataset(features, lead_hours=24)
    gen = _gen_history()
    X, y, cols = assemble_feature_matrix(day_ahead, gen, lead_hours=24)

    assert len(X) == len(day_ahead)
    assert len(y) == len(day_ahead)
    assert "hour" in cols
    assert "is_holiday" in cols
    assert "nwp_wind_speed_hub_mps" in cols
    assert WIND_DIRECTION_SIN_COL in cols
    assert AIR_DENSITY_COL in cols
    assert any(c.startswith(f"{TARGET_COL}_lag_") for c in cols)
    assert any(c.startswith(f"{TARGET_COL}_roll") for c in cols)
    assert TARGET_COL not in cols
    assert not any(c.startswith("era5_") for c in cols)


def test_validate_rejects_era5_features() -> None:
    df = _sample_features(n_hours=2)
    violations = validate_no_target_leakage(df, ["era5_wind_speed_hub_mps"], lead_hours=24)
    assert violations
    assert "era5" in violations[0].lower()


def test_validate_rejects_valid_time_lag_join_for_short_lags() -> None:
    """Simulate leakage: lag_1 merged at valid_time would read post-init target."""
    idx = pd.date_range("2019-06-01", periods=48, freq="h", tz="Europe/Berlin")
    gen = pd.Series(np.arange(48, dtype=float) * 10.0, index=idx, name=GENERATION_COL)
    vt = idx[-1]
    init = vt - pd.Timedelta(hours=24)
    leaked_value = float(gen.asof(vt - pd.Timedelta(hours=1)))

    df = pd.DataFrame(
        [
            {
                "valid_time": vt,
                "init_time": init,
                "lead_hours": 24,
                TARGET_COL: 1000.0,
                f"{TARGET_COL}_lag_1": leaked_value,
            }
        ]
    )
    violations = validate_no_target_leakage(
        df,
        [f"{TARGET_COL}_lag_1"],
        lead_hours=24,
        gen_history=gen,
    )
    assert violations
    assert "valid_time" in violations[0]


def test_assemble_passes_leakage_check_with_init_time_lags() -> None:
    """Proper init_time lag join must clear the day-ahead leakage validator."""
    features = _sample_features(n_hours=72)
    day_ahead = prepare_day_ahead_dataset(features, lead_hours=24)
    gen = _gen_history(n_hours=96)

    # validate=True runs init_time lag alignment checks inside assemble_feature_matrix.
    X, y, cols = assemble_feature_matrix(day_ahead, gen, lead_hours=24, validate=True)
    assert validate_no_target_leakage(day_ahead, cols, lead_hours=24) == []
    assert len(X) == len(y)
    assert X.attrs["lag_join_key"] == "init_time"
