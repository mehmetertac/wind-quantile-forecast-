"""Tests for OPSD wind download."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from wind_quantile_forecast.config import ENERGY_FP_RAW_DIR, WIND_TARGET_PARQUET
from wind_quantile_forecast.data.download import download_opsd_wind_data


@pytest.mark.skipif(
    not (ENERGY_FP_RAW_DIR / "opsd_time_series_60min.parquet").is_file(),
    reason="OPSD parquet not cached in energy-feature-pipeline",
)
def test_download_opsd_wind_data_writes_parquet(tmp_path: Path) -> None:
    out = download_opsd_wind_data(output_dir=tmp_path, country="DE")
    assert out == tmp_path / WIND_TARGET_PARQUET
    df = pd.read_parquet(out)
    assert "wind_mw" in df.columns
    assert len(df) > 1000
    assert df["wind_mw"].min() >= 0
