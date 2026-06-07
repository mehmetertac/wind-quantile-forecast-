"""Build day-ahead wind dataset: OPSD target + ERA5/NWP features (energy-feature-pipeline)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from energy_features.bias_correction import select_canonical_forecasts
from energy_features.calendar import make_calendar_features
from energy_features.data_loader import load_opsd_generation, resolve_opsd_parquet_path
from energy_features.generation_lift import feature_cols_for_set
from energy_features.pipeline import PlantMeta, _add_era5_physics, build_features
from energy_features.weather import (
    GFS_HISTORICAL_PRIORITY,
    ERA5Loader,
    extract_nwp_forecast_point,
    fetch_era5,
    resolve_era5_paths_for_range,
)

from wind_quantile_forecast.config import (
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    DAY_AHEAD_DATASET_PARQUET,
    DAY_AHEAD_LEAD_HOURS,
    DEFAULT_COUNTRY,
    DEFAULT_END,
    DEFAULT_START,
    DEFAULT_TIMEZONE,
    ENERGY_FP_RAW_DIR,
    FEATURES_PARQUET,
    GENERATION_COL,
    RANDOM_SEED,
    TARGET_COL,
)
from wind_quantile_forecast.config import (
    DEFAULT_LATITUDE as CFG_LAT,
)
from wind_quantile_forecast.config import (
    DEFAULT_LONGITUDE as CFG_LON,
)

NWP_INIT_FREQ = "12h"
OPERATIONAL_LEADS = [6, 12, 24, 48]


def resolve_era5_cache_dir(*, raw_dir: Path | None = None) -> Path:
    """Return directory containing monthly ERA5 NetCDF files."""
    raw_dir = raw_dir or DATA_RAW_DIR
    if any(raw_dir.glob("era5_*_de_bbox5deg_*.nc")):
        return raw_dir
    if any(ENERGY_FP_RAW_DIR.glob("era5_*_de_bbox5deg_*.nc")):
        return ENERGY_FP_RAW_DIR
    return raw_dir


def load_wind_generation(
    *,
    country: str = DEFAULT_COUNTRY,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    tz: str = DEFAULT_TIMEZONE,
    raw_dir: Path | None = None,
) -> pd.Series:
    """Load hourly DE wind generation (onshore + offshore) from OPSD."""
    cache_dir = raw_dir or DATA_RAW_DIR
    if resolve_opsd_parquet_path(cache_dir=cache_dir) is None:
        if resolve_opsd_parquet_path(cache_dir=ENERGY_FP_RAW_DIR) is not None:
            cache_dir = ENERGY_FP_RAW_DIR
    raw = load_opsd_generation(
        cache_dir=cache_dir,
        country=country,
        tech=("wind_onshore", "wind_offshore"),
    )
    gen = raw.sum(axis=1).astype(float).rename(GENERATION_COL)
    if gen.index.tz is None:
        gen.index = gen.index.tz_localize("UTC")
    return gen.tz_convert(tz).loc[start:end].dropna()


def ensure_era5_months(
    start: str,
    end: str,
    *,
    cache_dir: Path,
    tz: str = DEFAULT_TIMEZONE,
    download: bool = False,
) -> Path:
    """Verify ERA5 monthly files exist for ``[start, end]``; optionally download missing months."""
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    if start_ts.tz is None:
        start_ts = start_ts.tz_localize(tz)
    else:
        start_ts = start_ts.tz_convert(tz)
    if end_ts.tz is None:
        end_ts = end_ts.tz_localize(tz)
    else:
        end_ts = end_ts.tz_convert(tz)

    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        resolve_era5_paths_for_range(start_ts, end_ts, cache_dir=cache_dir)
        return cache_dir
    except FileNotFoundError:
        if not download:
            raise

    from energy_features.weather import _months_in_range

    for year, month in _months_in_range(start_ts, end_ts):
        path = cache_dir / f"era5_{year}_de_bbox5deg_{month:02d}.nc"
        if not path.is_file():
            fetch_era5(year=year, month=month, cache_dir=cache_dir)
    return cache_dir


def _fetch_nwp(
    start: str,
    end: str,
    leads: list[int],
    cache: Path,
    *,
    lat: float,
    lon: float,
    tz: str,
) -> pd.DataFrame | None:
    if cache.is_file():
        return pd.read_parquet(cache)

    inits = pd.date_range(start, end, freq=NWP_INIT_FREQ, tz=tz)
    parts: list[pd.DataFrame] = []
    for init in inits:
        try:
            part = extract_nwp_forecast_point(
                init,
                latitude=lat,
                longitude=lon,
                fxx=leads,
                priority=GFS_HISTORICAL_PRIORITY,
                tz=tz,
                verbose=False,
            )
            parts.append(part)
        except (FileNotFoundError, OSError, AssertionError):
            continue
    if not parts:
        return None
    nwp = pd.concat(parts).reset_index()
    cache.parent.mkdir(parents=True, exist_ok=True)
    nwp.to_parquet(cache)
    return nwp


def _synthetic_nwp_from_era5(hourly: pd.DataFrame, leads: list[int]) -> pd.DataFrame:
    """Long-format NWP rows from ERA5 truth + lead-dependent noise (offline fallback)."""
    rng = np.random.default_rng(RANDOM_SEED)
    base_cols = [
        "t2m_k",
        "t2m_c",
        "wind_speed_10m",
        "wind_direction_10m",
        "wind_u_10m",
        "wind_v_10m",
        "surface_solar_radiation_w_m2",
    ]
    rows: list[dict[str, object]] = []
    hourly_reset = hourly.reset_index(names="valid_time")
    for vt in hourly_reset["valid_time"]:
        for lead in leads:
            init = vt - pd.Timedelta(hours=int(lead))
            row: dict[str, object] = {
                "valid_time": vt,
                "init_time": init,
                "lead_hours": int(lead),
            }
            scale = 0.15 + 0.02 * float(lead)
            for col in base_cols:
                era5_key = f"era5_{col}"
                if era5_key not in hourly_reset.columns:
                    continue
                truth = float(hourly_reset.loc[hourly_reset["valid_time"] == vt, era5_key].iloc[0])
                if col == "wind_speed_10m":
                    row[col] = max(0.1, truth + scale * rng.normal(0.8, 1.0))
                elif col == "surface_solar_radiation_w_m2":
                    row[col] = max(0.0, truth + scale * 80.0 * rng.normal(0.0, 1.0))
                elif "direction" in col:
                    row[col] = truth + scale * 10.0 * rng.normal(0.0, 1.0)
                else:
                    row[col] = truth + scale * rng.normal(0.0, 1.0)
            rows.append(row)
    return pd.DataFrame(rows)


def build_wind_feature_table(
    *,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    country: str = DEFAULT_COUNTRY,
    lat: float = CFG_LAT,
    lon: float = CFG_LON,
    tz: str = DEFAULT_TIMEZONE,
    raw_dir: Path | None = None,
    allow_synthetic_nwp: bool = True,
    download_era5: bool = False,
) -> pd.DataFrame:
    """Join OPSD wind generation with ERA5, GFS NWP, calendar, and wind physics features."""
    raw_dir = raw_dir or DATA_RAW_DIR
    era5_dir = ensure_era5_months(
        start,
        end,
        cache_dir=resolve_era5_cache_dir(raw_dir=raw_dir),
        tz=tz,
        download=download_era5,
    )

    gen = load_wind_generation(country=country, start=start, end=end, tz=tz, raw_dir=raw_dir)
    meta = PlantMeta(latitude=lat, longitude=lon, tech="wind", tz=tz, country=country)

    nwp_path = raw_dir / f"nwp_gfs_{start}_{end}_{NWP_INIT_FREQ}.parquet"
    nwp = _fetch_nwp(start, end, OPERATIONAL_LEADS, nwp_path, lat=lat, lon=lon, tz=tz)
    if nwp is None:
        if not allow_synthetic_nwp:
            raise RuntimeError("GFS NWP fetch failed and synthetic fallback is disabled")
        era5 = ERA5Loader(cache_dir=era5_dir, tz=tz).load(lat, lon, start, end, tz=tz)
        hourly = gen.to_frame().join(era5.add_prefix("era5_"), how="inner")
        hourly = hourly.join(make_calendar_features(hourly.index, country=country))
        hourly = _add_era5_physics(hourly, meta)
        nwp = _synthetic_nwp_from_era5(hourly, OPERATIONAL_LEADS)

    if not nwp_path.is_file():
        nwp_path.parent.mkdir(parents=True, exist_ok=True)
        nwp.to_parquet(nwp_path)

    return build_features(gen, meta, era5_dir, nwp_path)


def prepare_day_ahead_dataset(
    features: pd.DataFrame,
    *,
    lead_hours: int = DAY_AHEAD_LEAD_HOURS,
    feature_set: str = "calendar_nwp_raw",
) -> pd.DataFrame:
    """Filter to day-ahead lead and attach modeling feature column list metadata.

    Args:
        features: Long-format table from :func:`build_wind_feature_table`.
        lead_hours: Operational forecast horizon (default 24 h day-ahead).
        feature_set: Named feature ablation from energy-feature-pipeline.

    Returns:
        One row per ``valid_time`` at ``lead_hours`` with target ``wind_mw``.
    """
    at_lead = features.loc[features["lead_hours"] == lead_hours].copy()
    if at_lead.empty:
        msg = f"no rows at lead_hours={lead_hours}"
        raise ValueError(msg)

    out = select_canonical_forecasts(at_lead)
    out = out.rename(columns={GENERATION_COL: TARGET_COL})
    out.attrs["feature_cols"] = feature_cols_for_set(out, feature_set)
    out.attrs["lead_hours"] = lead_hours
    out.attrs["feature_set"] = feature_set
    return out.reset_index(drop=True)


def pull_day_ahead_dataset(
    *,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    country: str = DEFAULT_COUNTRY,
    raw_dir: Path | None = None,
    processed_dir: Path | None = None,
    allow_synthetic_nwp: bool = True,
    download_era5: bool = False,
    lead_hours: int = DAY_AHEAD_LEAD_HOURS,
) -> tuple[Path, Path]:
    """End-to-end: OPSD wind + ERA5/NWP features → processed day-ahead parquet.

    Returns:
        Tuple of (long features path, day-ahead modeling path).
    """
    raw_dir = raw_dir or DATA_RAW_DIR
    processed_dir = processed_dir or DATA_PROCESSED_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    features = build_wind_feature_table(
        start=start,
        end=end,
        country=country,
        raw_dir=raw_dir,
        allow_synthetic_nwp=allow_synthetic_nwp,
        download_era5=download_era5,
    )
    features_path = processed_dir / FEATURES_PARQUET
    features.to_parquet(features_path, index=False)

    day_ahead = prepare_day_ahead_dataset(features, lead_hours=lead_hours)
    day_ahead_path = processed_dir / DAY_AHEAD_DATASET_PARQUET
    day_ahead.to_parquet(day_ahead_path, index=False)
    return features_path, day_ahead_path


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for dataset pull."""
    parser = argparse.ArgumentParser(
        description="Pull OPSD wind + ERA5/NWP day-ahead feature dataset.",
    )
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--country", default=DEFAULT_COUNTRY)
    parser.add_argument("--lead-hours", type=int, default=DAY_AHEAD_LEAD_HOURS)
    parser.add_argument(
        "--download-era5",
        action="store_true",
        help="Download missing ERA5 months from Copernicus CDS (needs ~/.cdsapirc)",
    )
    parser.add_argument(
        "--no-synthetic-nwp",
        action="store_true",
        help="Fail if GFS/Herbie NWP cannot be fetched",
    )
    args = parser.parse_args(argv)

    features_path, day_ahead_path = pull_day_ahead_dataset(
        start=args.start,
        end=args.end,
        country=args.country,
        allow_synthetic_nwp=not args.no_synthetic_nwp,
        download_era5=args.download_era5,
        lead_hours=args.lead_hours,
    )
    day_ahead = pd.read_parquet(day_ahead_path)
    n_rows = len(day_ahead)
    n_feats = len(day_ahead.attrs.get("feature_cols", []))
    print(f"Wrote {features_path}")
    print(f"Wrote {day_ahead_path} — {n_rows} hours, lead={args.lead_hours}h, {n_feats} features")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
