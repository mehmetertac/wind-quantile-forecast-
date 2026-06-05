"""Clean and align wind generation time series."""

from pathlib import Path

import pandas as pd

from wind_quantile_forecast.config import DATA_PROCESSED_DIR


def preprocess_wind_data(
    raw_path: Path,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Load raw OPSD data, clean missing values, and resample to hourly.

    Args:
        raw_path: Path to the raw CSV downloaded from OPSD.
        output_dir: Directory to save processed parquet. Defaults to data/processed/.

    Returns:
        Processed DataFrame with datetime index and wind generation column.

    Raises:
        NotImplementedError: Preprocessing logic not yet implemented.
    """
    output_dir = output_dir or DATA_PROCESSED_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    raise NotImplementedError(f"Preprocessing not yet implemented for {raw_path}")
