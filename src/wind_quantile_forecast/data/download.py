"""Download wind generation data from Open Power System Data (OPSD)."""

from pathlib import Path

from wind_quantile_forecast.config import DATA_RAW_DIR, OPSD_BASE_URL, OPSD_WIND_DATASET


def download_opsd_wind_data(
    output_dir: Path | None = None,
    country: str = "DE",
) -> Path:
    """Download OPSD wind generation time series for a given country.

    Args:
        output_dir: Directory to save raw CSV files. Defaults to data/raw/.
        country: ISO country code (e.g. "DE" for Germany).

    Returns:
        Path to the downloaded CSV file.

    Raises:
        NotImplementedError: Download logic not yet implemented.
    """
    output_dir = output_dir or DATA_RAW_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    raise NotImplementedError(
        f"OPSD download not yet implemented. "
        f"Target: {OPSD_BASE_URL}/{OPSD_WIND_DATASET}, country={country}"
    )
