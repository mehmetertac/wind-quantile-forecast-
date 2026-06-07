"""Project configuration: paths, quantiles, and reproducibility."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports" / "figures"

QUANTILES = [0.1, 0.5, 0.9]
QUANTILE_LABELS = {0.1: "P10", 0.5: "P50", 0.9: "P90"}

RANDOM_SEED = 42

# Open Power System Data
OPSD_BASE_URL = "https://data.open-power-system-data.org"
OPSD_WIND_DATASET = "time_series"
OPSD_PARQUET_NAME = "opsd_time_series_60min.parquet"
WIND_TARGET_PARQUET = "wind_generation_hourly.parquet"
DAY_AHEAD_DATASET_PARQUET = "day_ahead_wind.parquet"
FEATURES_PARQUET = "wind_features_long.parquet"

# Germany study point (matches energy-feature-pipeline)
DEFAULT_LATITUDE = 52.5
DEFAULT_LONGITUDE = 13.4
DEFAULT_TIMEZONE = "Europe/Berlin"
DEFAULT_COUNTRY = "DE"

# Day-ahead forecast: predict hourly wind at valid_time using NWP issued ~24 h earlier
DAY_AHEAD_LEAD_HOURS = 24
DEFAULT_START = "2019-06-01"
DEFAULT_END = "2019-06-15"

TARGET_COL = "wind_mw"
GENERATION_COL = "generation_mw"

# Autoregressive features (joined at init_time — safe for day-ahead lead=24h)
DEFAULT_LAGS: tuple[int, ...] = (1, 6, 12, 24, 48)
DEFAULT_ROLL_WINDOWS: tuple[int, ...] = (24, 168)

# GFS NWP merge schema has no surface pressure; use MSLP fallback for air density
NWP_STANDARD_PRESSURE_PA = 101_325.0

# Rolling-origin evaluation (expanding train window by default)
DEFAULT_CV_FOLDS = 5
VALID_TIME_COL = "valid_time"

# Sibling energy-feature-pipeline cache (OPSD parquet, ERA5 NetCDF, NWP parquet)
ENERGY_FP_ROOT = PROJECT_ROOT.parent / "energy-feature-pipeline" / "energy-feature-pipeline"
ENERGY_FP_RAW_DIR = ENERGY_FP_ROOT / "data" / "raw"
