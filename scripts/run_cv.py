"""Run quantile GBM rolling-origin CV and write results/metrics.csv."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from wind_quantile_forecast.config import (
    DEFAULT_CV_FOLDS,
    HOUR_SEASON_COL,
    METRICS_CSV,
)
from wind_quantile_forecast.data import build_modeling_table
from wind_quantile_forecast.evaluation import evaluate_quantile_origin_cv
from wind_quantile_forecast.features.target_encoding import CatEncoding
from wind_quantile_forecast.models import make_quantile_predict_fold


def main(argv: list[str] | None = None) -> int:
    """Run quantile GBM CV and save per-fold metrics to CSV."""
    parser = argparse.ArgumentParser(description="Rolling-origin CV → results/metrics.csv")
    parser.add_argument("--n-splits", type=int, default=DEFAULT_CV_FOLDS)
    parser.add_argument("--metrics-path", type=str, default=str(METRICS_CSV))
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument(
        "--backend",
        choices=["lightgbm", "xgboost", "catboost"],
        default="lightgbm",
        help="GBM backend for quantile regression (default: lightgbm)",
    )
    parser.add_argument(
        "--per-quantile",
        action="store_true",
        help="Train one model per quantile (XGBoost/CatBoost default: multi-quantile)",
    )
    parser.add_argument(
        "--cat-encoding",
        choices=["none", "target", "native"],
        default="target",
        help="Encoding for hour_season (default: target; native=CatBoost only)",
    )
    parser.add_argument(
        "--params-json",
        type=str,
        default=None,
        help="Load model_params from final_model_params.json (or similar)",
    )
    parser.add_argument(
        "--no-monotonic",
        action="store_true",
        help="Disable P10<=P50<=P90 post-processing",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    model_df, feature_cols = build_modeling_table()
    model_params: dict[str, int | float] = {"n_estimators": args.n_estimators}
    if args.params_json:
        payload = json.loads(Path(args.params_json).read_text(encoding="utf-8"))
        model_params = dict(payload.get("model_params", model_params))
        if "backend" in payload:
            args.backend = payload["backend"]
        if "cat_encoding" in payload:
            args.cat_encoding = payload["cat_encoding"]
    if args.backend == "lightgbm":
        model_params.setdefault("verbosity", -1)
    cat_encoding: CatEncoding = args.cat_encoding
    if cat_encoding == "native" and args.backend != "catboost":
        parser.error("--cat-encoding native requires --backend catboost")
    predict_fold = make_quantile_predict_fold(
        feature_cols,
        backend=args.backend,
        model_params=model_params,
        multi_quantile=not args.per_quantile,
        cat_col=HOUR_SEASON_COL,
        cat_encoding=cat_encoding,
        enforce_monotonic=not args.no_monotonic,
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
