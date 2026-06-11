"""Compare LightGBM / XGBoost / CatBoost on probabilistic metrics + training time."""

from __future__ import annotations

import argparse
import logging

from wind_quantile_forecast.config import (
    COMPARISON_CSV,
    DEFAULT_CV_FOLDS,
    HOUR_SEASON_COL,
)
from wind_quantile_forecast.data import build_modeling_table
from wind_quantile_forecast.evaluation.comparison import (
    comparison_table_markdown,
    run_backend_comparison,
)


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
