"""Run LightGBM rolling-origin CV and write results/metrics.csv."""

from __future__ import annotations

import argparse
import logging

import pandas as pd
from wind_quantile_forecast.config import (
    DAY_AHEAD_LEAD_HOURS,
    DEFAULT_CV_FOLDS,
    METRICS_CSV,
    TARGET_COL,
    VALID_TIME_COL,
)
from wind_quantile_forecast.data import load_day_ahead_dataset, load_wind_generation
from wind_quantile_forecast.evaluation import evaluate_quantile_origin_cv
from wind_quantile_forecast.features import assemble_feature_matrix
from wind_quantile_forecast.features.lags import LEAD_COL
from wind_quantile_forecast.models import make_quantile_predict_fold


def build_modeling_table() -> tuple[pd.DataFrame, list[str]]:
    """Load day-ahead data and return a CV-ready frame with feature columns."""
    day_ahead = load_day_ahead_dataset()
    gen_history = load_wind_generation()
    at_lead = day_ahead.loc[day_ahead[LEAD_COL] == DAY_AHEAD_LEAD_HOURS]
    X, y, feature_cols = assemble_feature_matrix(day_ahead, gen_history)
    model_df = X.copy()
    model_df[TARGET_COL] = y
    model_df[VALID_TIME_COL] = at_lead.loc[X.index, VALID_TIME_COL]
    return model_df, feature_cols


def main(argv: list[str] | None = None) -> int:
    """Run quantile GBM CV and save per-fold metrics to CSV."""
    parser = argparse.ArgumentParser(description="Rolling-origin CV → results/metrics.csv")
    parser.add_argument("--n-splits", type=int, default=DEFAULT_CV_FOLDS)
    parser.add_argument("--metrics-path", type=str, default=str(METRICS_CSV))
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    model_df, feature_cols = build_modeling_table()
    predict_fold = make_quantile_predict_fold(
        feature_cols,
        model_params={"n_estimators": args.n_estimators, "verbosity": -1},
    )
    result = evaluate_quantile_origin_cv(
        model_df,
        feature_cols,
        predict_fold,
        n_splits=args.n_splits,
        metrics_path=args.metrics_path,
    )
    print(f"Wrote {args.metrics_path} ({len(result.fold_metrics)} folds)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
