# wind-quantile-forecast

Day-ahead probabilistic wind power forecast producing **P10 / P50 / P90** quantiles, evaluated with pinball loss and reliability diagrams.

## Why it matters

Most utility production forecasting still runs on quantile gradient boosting models, not deep learning. This project covers the full production stack: LightGBM, XGBoost, CatBoost, pinball loss, SHAP interpretability, and target encoding for high-cardinality features.

## Problem

Given historical wind generation and weather features, predict the next-day hourly wind power distribution as three quantiles:

| Quantile | Label | Meaning |
|----------|-------|---------|
| 0.10 | P10 | 10 % chance actual generation falls below this value |
| 0.50 | P50 | Median (point) forecast |
| 0.90 | P90 | 90 % chance actual generation falls below this value |

## Data source

**Target:** German hourly wind power (OPSD onshore + offshore actual generation).

**Features:** Reuses the [energy-feature-pipeline](https://github.com/mehmetertac/energy-feature-pipeline) ERA5 reanalysis + GFS NWP stack (calendar, wind physics, hub-height extrapolation).

| Layer | Source | Role |
|-------|--------|------|
| Generation | [OPSD](https://open-power-system-data.org/) time series | `wind_mw` target |
| Reanalysis | Copernicus ERA5 (CDS) | Hindcast weather + physics features |
| NWP | GFS via Herbie | Day-ahead forecast covariates at 24 h lead |

Install the sibling feature pipeline, then pull the dataset:

```powershell
pip install -r requirements-dev.txt
pip install -r requirements-weather.txt   # editable install of ../energy-feature-pipeline
pip install -e .

# OPSD wind + ERA5/NWP features → data/processed/day_ahead_wind.parquet
pull-wind-data --start 2019-06-01 --end 2019-06-15

# Download missing ERA5 months from Copernicus (needs ~/.cdsapirc)
pull-wind-data --download-era5
```

The sibling project's cached OPSD parquet is used automatically when present at
`../energy-feature-pipeline/energy-feature-pipeline/data/raw/`.

## Tech stack

- **Models:** LightGBM, XGBoost, CatBoost (quantile regression via pinball loss)
- **Interpretability:** SHAP
- **Encoding:** Target encoding for high-cardinality categorical features
- **Evaluation:** Pinball loss, coverage metrics, reliability (calibration) diagrams

## Repository layout

```
wind-quantile-forecast/
├── src/wind_quantile_forecast/
│   ├── config.py              # paths, quantiles, seeds
│   ├── data/                  # OPSD download + preprocessing
│   ├── features/              # calendar, lags, target encoding
│   ├── models/                # pinball loss + QuantileGBM wrapper
│   ├── evaluation/            # metrics, reliability, plots
│   ├── interpret/             # SHAP explanations
│   └── cli.py                 # pipeline entry point
├── tests/
│   ├── unit/                  # unit tests (pinball loss, etc.)
│   └── integration/           # end-to-end pipeline tests
├── data/{raw,processed}/      # data directories (contents gitignored)
├── reports/figures/           # calibration plots + SHAP output
├── notebooks/                 # exploratory analysis
├── scripts/                   # pre-commit helpers
├── AGENTS.md                  # contributor / agent governance rules
└── requirements.txt
```

## Pipeline

```mermaid
flowchart LR
    OPSD["OPSD wind\n(onshore+offshore)"] --> Download
    ERA5["ERA5 reanalysis"] --> Features
    NWP["GFS NWP"] --> Features
    Download --> Preprocess
    Preprocess --> Features["energy-feature-pipeline\nfeatures"]
    Features --> DayAhead["Day-ahead table\nlead=24h"]
    DayAhead --> Train["Quantile GBM\nLightGBM / XGBoost / CatBoost"]
    Train --> Predict["P10 / P50 / P90"]
    Predict --> Eval["Pinball loss +\nReliability diagrams"]
    Train --> SHAP["SHAP explanations"]
    Eval --> Reports["reports/figures/"]
    SHAP --> Reports
```

## Setup

```powershell
# Clone and enter the repo
git clone https://github.com/mehmetertac/wind-quantile-forecast-.git
cd wind-quantile-forecast-

# Create virtual environment
py -m venv .venv
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements-dev.txt
pip install -e .

# Install pre-commit hooks (runs tests + file-size check before commit)
pre-commit install
```

## Usage

```powershell
# Run the full pipeline (not yet implemented)
wind-forecast --country DE --backend lightgbm

# Run tests
pytest

# Run linting
ruff check src tests
```

## Deliverables

- [x] OPSD data ingestion and preprocessing
- [x] ERA5/NWP feature reuse from energy-feature-pipeline (day-ahead lead=24h)
- [x] Feature engineering (calendar, lags, weather drivers, leakage-safe matrix)
- [ ] Quantile GBM models (LightGBM, XGBoost, CatBoost)
- [x] Rolling-origin CV harness (`RollingOriginSplit`, `run_rolling_origin_cv`)
- [ ] Pinball loss evaluation (fold metrics helper ready; full pipeline pending)
- [ ] Reliability / calibration diagrams
- [ ] SHAP feature importance plots

## License

Apache-2.0 — see [LICENSE](LICENSE).
