"""Compare LightGBM / XGBoost / CatBoost on probabilistic metrics + training time."""

from __future__ import annotations

import argparse
import logging

import pandas as pd
from wind_quantile_forecast.config import (
    COMPARISON_CSV,
    DAY_AHEAD_LEAD_HOURS,
    DEFAULT_CV_FOLDS,
    HOUR_SEASON_COL,
    TARGET_COL,
    VALID_TIME_COL,
)
from wind_quantile_forecast.data import load_day_ahead_dataset, load_wind_generation
from wind_quantile_forecast.evaluation.comparison import (
    comparison_table_markdown,
    run_backend_comparison,
)
from wind_quantile_forecast.features import assemble_feature_matrix
from wind_quantile_forecast.features.lags import LEAD_COL


def build_modeling_table() -> tuple[pd.DataFrame, list[str]]:
    """Load day-ahead data with ``hour_season`` high-cardinality feature."""
    day_ahead = load_day_ahead_dataset()
    gen_history = load_wind_generation()
    at_lead = day_ahead.loc[day_ahead[LEAD_COL] == DAY_AHEAD_LEAD_HOURS]
    X, y, feature_cols = assemble_feature_matrix(
        day_ahead,
        gen_history,
        include_high_cardinality=True,
    )
    model_df = X.copy()
    model_df[TARGET_COL] = y
    model_df[VALID_TIME_COL] = at_lead.loc[X.index, VALID_TIME_COL]
    return model_df, feature_cols


def main(argv: list[str] | None = None) -> int:
    """Run backend comparison CV and write ``results/backend_comparison.csv``."""
    parser = argparse.ArgumentParser(
        description="Compare quantile GBM backends on probabilistic metrics + train time"
    )
    parser.add_argument("--n-splits", type=int, default=DEFAULT_CV_FOLDS)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument(
        "--comparison-path",
        type=str,
        default=str(COMPARISON_CSV),
        help="Output CSV path (default: results/backend_comparison.csv)",
    )
    parser.add_argument(
        "--per-quantile",
        action="store_true",
        help="Train one model per quantile for XGBoost/CatBoost",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    model_df, feature_cols = build_modeling_table()
    if HOUR_SEASON_COL not in feature_cols:
        msg = f"{HOUR_SEASON_COL!r} missing from feature columns"
        raise RuntimeError(msg)

    table = run_backend_comparison(
        model_df,
        feature_cols,
        cat_col=HOUR_SEASON_COL,
        n_splits=args.n_splits,
        model_params={"n_estimators": args.n_estimators, "verbosity": -1},
        multi_quantile=not args.per_quantile,
        comparison_path=args.comparison_path,
    )
    print(comparison_table_markdown(table))
    print(f"\nWrote {args.comparison_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
