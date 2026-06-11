"""Fit best quantile GBM and write SHAP summary + dependence plots."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from wind_quantile_forecast.config import (
    HOUR_SEASON_COL,
    QUANTILES,
    REPORTS_DIR,
    TARGET_COL,
)
from wind_quantile_forecast.data import build_modeling_table
from wind_quantile_forecast.features.target_encoding import (
    add_target_encoding,
    target_encoding_column,
)
from wind_quantile_forecast.interpret.shap_explain import (
    DEFAULT_DEPENDENCE_FEATURES,
    explain_model,
)
from wind_quantile_forecast.models.quantile_gbm import Backend, QuantileGBM


def _resolve_model_columns(feature_cols: list[str]) -> list[str]:
    """Replace ``hour_season`` with its target-encoded column name."""
    te_col = target_encoding_column(HOUR_SEASON_COL)
    return [te_col if c == HOUR_SEASON_COL else c for c in feature_cols]


def main(argv: list[str] | None = None) -> int:
    """Fit final quantile model(s) and generate SHAP plots."""
    parser = argparse.ArgumentParser(
        description="Fit quantile GBM and write SHAP plots to reports/figures/",
    )
    parser.add_argument(
        "--backend",
        choices=["lightgbm", "xgboost", "catboost"],
        default="lightgbm",
        help="GBM backend (default: lightgbm — best on backend comparison)",
    )
    parser.add_argument(
        "--quantiles",
        type=float,
        nargs="+",
        default=[0.5, 0.9],
        help="Quantile levels to explain (default: 0.5 0.9)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPORTS_DIR),
        help="Directory for SHAP PNG output",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=500,
        help="Max rows for SHAP computation",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="GBM tree count for final fit",
    )
    parser.add_argument(
        "--dependence-features",
        nargs="+",
        default=list(DEFAULT_DEPENDENCE_FEATURES),
        help="Features for dependence plots",
    )
    parser.add_argument(
        "--per-quantile",
        action="store_true",
        help="Train one model per quantile (required for SHAP on XGB/CatBoost)",
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

    encoded_df = add_target_encoding(model_df, HOUR_SEASON_COL, TARGET_COL)
    model_cols = _resolve_model_columns(feature_cols)

    backend: Backend = args.backend
    multi_quantile = not args.per_quantile
    if backend != "lightgbm" and multi_quantile:
        parser.error(
            "SHAP requires per-quantile estimators; pass --per-quantile for "
            "xgboost/catboost or use --backend lightgbm"
        )

    model_params: dict[str, int] = {"n_estimators": args.n_estimators}
    if backend == "lightgbm":
        model_params["verbosity"] = -1

    quantiles = list(args.quantiles)
    gbm = QuantileGBM(
        backend=backend,
        quantiles=quantiles if args.per_quantile else QUANTILES,
        model_params=model_params,
        multi_quantile=multi_quantile,
    )
    gbm.fit(encoded_df[model_cols], encoded_df[TARGET_COL])

    output_dir = Path(args.output_dir)
    written: list[Path] = []
    for q in quantiles:
        out = explain_model(
            gbm,
            encoded_df[model_cols],
            quantile=q,
            dependence_features=args.dependence_features,
            output_dir=output_dir,
            max_samples=args.max_samples,
        )
        label = f"p{int(q * 100)}"
        written.extend(sorted(out.glob(f"shap_*_{label}*.png")))

    print(f"Wrote {len(written)} SHAP plots to {output_dir}")
    for path in sorted(written):
        print(f"  {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
