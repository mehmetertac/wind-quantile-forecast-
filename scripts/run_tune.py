"""Optuna hyperparameter tuning for LightGBM quantile models."""

from __future__ import annotations

import argparse
import logging

from wind_quantile_forecast.config import DEFAULT_CV_FOLDS, FINAL_PARAMS_JSON, TUNING_CSV
from wind_quantile_forecast.data import build_modeling_table
from wind_quantile_forecast.evaluation.tuning import (
    default_final_config,
    save_final_model_params,
    save_tuning_trials,
    tune_lightgbm_quantile_cv,
)


def main(argv: list[str] | None = None) -> int:
    """Run Optuna tuning and write trials CSV + final params JSON."""
    parser = argparse.ArgumentParser(
        description="Tune LightGBM on rolling-origin CV pinball loss (Optuna)",
    )
    parser.add_argument("--n-trials", type=int, default=40)
    parser.add_argument("--n-splits", type=int, default=DEFAULT_CV_FOLDS)
    parser.add_argument("--trials-path", type=str, default=str(TUNING_CSV))
    parser.add_argument("--params-path", type=str, default=str(FINAL_PARAMS_JSON))
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    model_df, feature_cols = build_modeling_table()
    result = tune_lightgbm_quantile_cv(
        model_df,
        feature_cols,
        n_trials=args.n_trials,
        n_splits=args.n_splits,
    )
    save_tuning_trials(result.trials, args.trials_path)
    config = default_final_config(result.best_params)
    config["tuning"] = {
        "best_mean_pinball": result.best_score,
        "n_trials": args.n_trials,
    }
    save_final_model_params(config, path=args.params_path)
    print(f"Best mean pinball: {result.best_score:.4f}")
    print(f"Wrote {args.trials_path}")
    print(f"Wrote {args.params_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
