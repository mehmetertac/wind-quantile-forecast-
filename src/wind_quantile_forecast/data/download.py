"""Download wind generation data from Open Power System Data (OPSD)."""

from __future__ import annotations

import shutil
from pathlib import Path

from energy_features.data_loader import (
    DEFAULT_PARQUET_NAME,
    load_opsd_generation,
    resolve_opsd_parquet_path,
)

from wind_quantile_forecast.config import (
    DATA_RAW_DIR,
    ENERGY_FP_RAW_DIR,
    OPSD_BASE_URL,
    OPSD_WIND_DATASET,
    WIND_TARGET_PARQUET,
)


def _resolve_opsd_cache_dir(output_dir: Path) -> Path:
    """Prefer sibling energy-feature-pipeline cache when local OPSD parquet is absent."""
    local = resolve_opsd_parquet_path(cache_dir=output_dir)
    if local is not None:
        return output_dir
    sibling = resolve_opsd_parquet_path(cache_dir=ENERGY_FP_RAW_DIR)
    if sibling is not None:
        return ENERGY_FP_RAW_DIR
    return output_dir


def download_opsd_wind_data(
    output_dir: Path | None = None,
    country: str = "DE",
    *,
    force_download: bool = False,
) -> Path:
    """Download or cache OPSD wind generation time series for a given country.

    Loads onshore + offshore actual generation via ``energy_features`` (shared with
    energy-feature-pipeline), sums to a single hourly wind target, and writes
    ``wind_generation_hourly.parquet``.

    Args:
        output_dir: Directory to save raw parquet files. Defaults to data/raw/.
        country: ISO country code (e.g. "DE" for Germany).
        force_download: Re-fetch OPSD CSV even when a cached parquet exists.

    Returns:
        Path to the wind target parquet file.
    """
    output_dir = output_dir or DATA_RAW_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / WIND_TARGET_PARQUET

    cache_dir = _resolve_opsd_cache_dir(output_dir)
    raw = load_opsd_generation(
        cache_dir=cache_dir,
        country=country,
        tech=("wind_onshore", "wind_offshore"),
        force_download=force_download,
    )
    wind = raw.sum(axis=1).astype("float64").rename("wind_mw")
    wind.index.name = "timestamp"
    wind.to_frame().to_parquet(out_path)

    opsd_cache = cache_dir / DEFAULT_PARQUET_NAME
    local_opsd = output_dir / DEFAULT_PARQUET_NAME
    if opsd_cache.is_file() and not local_opsd.is_file() and cache_dir != output_dir:
        shutil.copy2(opsd_cache, local_opsd)

    return out_path


def opsd_source_hint() -> str:
    """Human-readable OPSD source string for logging and errors."""
    return f"{OPSD_BASE_URL}/{OPSD_WIND_DATASET}"
