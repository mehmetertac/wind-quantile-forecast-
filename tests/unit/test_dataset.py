"""Tests for day-ahead dataset preparation."""

from __future__ import annotations

import pandas as pd
import pytest
from wind_quantile_forecast.config import GENERATION_COL, TARGET_COL
from wind_quantile_forecast.data.dataset import prepare_day_ahead_dataset


def _sample_features(n_hours: int = 4, leads: list[int] | None = None) -> pd.DataFrame:
    leads = leads or [24, 48]
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
                    GENERATION_COL: 1000.0 + vt.hour * 10,
                    "hour": vt.hour,
                    "day_of_week": vt.dayofweek,
                    "month": vt.month,
                    "nwp_wind_speed_10m": 5.0 + vt.hour * 0.1,
                    "nwp_wind_speed_hub_mps": 6.0 + vt.hour * 0.1,
                }
            )
    return pd.DataFrame(rows)


def test_prepare_day_ahead_filters_lead_24() -> None:
    features = _sample_features()
    out = prepare_day_ahead_dataset(features, lead_hours=24)
    assert (out["lead_hours"] == 24).all()
    assert len(out) == features["valid_time"].nunique()
    assert TARGET_COL in out.columns
    assert GENERATION_COL not in out.columns


def test_prepare_day_ahead_missing_lead_raises() -> None:
    features = _sample_features(leads=[6, 12])
    with pytest.raises(ValueError, match="no rows at lead_hours=24"):
        prepare_day_ahead_dataset(features, lead_hours=24)


def test_prepare_day_ahead_feature_cols_attr() -> None:
    features = _sample_features()
    out = prepare_day_ahead_dataset(features, lead_hours=24)
    cols = out.attrs.get("feature_cols")
    assert isinstance(cols, list)
    assert "hour" in cols
    assert any(c.startswith("nwp_") for c in cols)
