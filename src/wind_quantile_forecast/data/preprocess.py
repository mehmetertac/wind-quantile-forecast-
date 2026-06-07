"""Clean and align wind generation time series."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from wind_quantile_forecast.config import (
    DATA_PROCESSED_DIR,
    DAY_AHEAD_DATASET_PARQUET,
    DEFAULT_TIMEZONE,
    TARGET_COL,
)
from wind_quantile_forecast.data.download import download_opsd_wind_data


def preprocess_wind_data(
    raw_path: Path,
    output_dir: Path | None = None,
    *,
    tz: str = DEFAULT_TIMEZONE,
) -> pd.DataFrame:
    """Load raw OPSD wind parquet, clean missing values, and resample to hourly.

    Args:
        raw_path: Path to ``wind_generation_hourly.parquet`` from OPSD download.
        output_dir: Directory to save processed parquet. Defaults to data/processed/.
        tz: Target timezone for the datetime index.

    Returns:
        Processed DataFrame with datetime index and ``wind_mw`` column.
    """
    output_dir = output_dir or DATA_PROCESSED_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(raw_path)
    if TARGET_COL not in df.columns:
        msg = f"expected column {TARGET_COL!r} in {raw_path}"
        raise KeyError(msg)

    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        df.index = pd.to_datetime(df.index, utc=True)

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(tz)

    out = df[[TARGET_COL]].astype("float64").sort_index()
    out = out.asfreq("1h")
    out[TARGET_COL] = out[TARGET_COL].interpolate(limit_direction="both")
    out = out.dropna()

    out_path = output_dir / raw_path.name
    out.to_parquet(out_path)
    return out


def load_day_ahead_dataset(
    processed_dir: Path | None = None,
) -> pd.DataFrame:
    """Load the day-ahead modeling table built by :func:`pull_day_ahead_dataset`."""
    processed_dir = processed_dir or DATA_PROCESSED_DIR
    path = processed_dir / DAY_AHEAD_DATASET_PARQUET
    if not path.is_file():
        msg = f"day-ahead dataset not found at {path}; run pull_dataset first"
        raise FileNotFoundError(msg)
    return pd.read_parquet(path)


def pull_and_preprocess_wind(
    output_dir: Path | None = None,
    country: str = "DE",
) -> pd.DataFrame:
    """Download OPSD wind and return hourly cleaned target series."""
    raw_path = download_opsd_wind_data(output_dir=output_dir, country=country)
    return preprocess_wind_data(raw_path, output_dir=output_dir)
