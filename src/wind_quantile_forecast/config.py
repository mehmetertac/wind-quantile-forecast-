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
