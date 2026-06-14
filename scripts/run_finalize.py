"""Finalize tuned model: CV, calibration plots, and locked params."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from wind_quantile_forecast.config import (
    CALIBRATION_CSV,
    DEFAULT_CV_FOLDS,
    FINAL_PARAMS_JSON,
    HOUR_SEASON_COL,
    METRICS_CSV,
    RELIABILITY_PNG,
    RESULTS_FIGURES_DIR,
    TUNING_CSV,
)
from wind_quantile_forecast.data import build_modeling_table
from wind_quantile_forecast.evaluation import (
    build_calibration_table,
    evaluate_quantile_origin_cv,
    mean_pinball_cv_score,
    pi_coverage,
    plot_reliability_diagram,
    reliability_from_oof,
)
from wind_quantile_forecast.evaluation.metrics import PI_COVERAGE_TARGET
from wind_quantile_forecast.evaluation.tuning import (
    default_final_config,
    load_final_model_params,
    save_final_model_params,
    save_tuning_trials,
    tune_lightgbm_quantile_cv,
)
from wind_quantile_forecast.features.target_encoding import CatEncoding
from wind_quantile_forecast.models import make_quantile_predict_fold

logger = logging.getLogger(__name__)


def _cv_summary(fold_metrics: pd.DataFrame) -> dict[str, float]:
    """Summarize key CV metrics from per-fold table."""
    folds = fold_metrics[fold_metrics["fold"] > 0]
    summary: dict[str, float] = {"mean_pinball": mean_pinball_cv_score(fold_metrics)}
    for col in ("pinball_q10", "pinball_q50", "pinball_q90", "pi_coverage", "mae", "rmse"):
        if col in folds.columns:
            summary[col] = float(folds[col].mean())
    return summary


def run_calibration_artifacts(
    oof: pd.DataFrame,
    *,
    calibration_path: Path = CALIBRATION_CSV,
    reliability_path: Path = RELIABILITY_PNG,
) -> None:
    """Write calibration CSV and reliability diagram from OOF predictions.

    Args:
        oof: Out-of-fold prediction DataFrame.
        calibration_path: Path for calibration coverage CSV.
        reliability_path: Path for reliability diagram PNG.
    """
    table = build_calibration_table(oof)
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(calibration_path, index=False, float_format="%.6f")
    logger.info("saved calibration table to %s", calibration_path)

    nominal, observed = reliability_from_oof(oof)
    pi_row = table.loc[table["metric"] == "pi_interval"]
    pi_cov = float(pi_row["observed"].iloc[0]) if not pi_row.empty else None
    plot_reliability_diagram(
        nominal,
        observed,
        reliability_path,
        pi_coverage=pi_cov,
        pi_target=PI_COVERAGE_TARGET,
    )
    logger.info("saved reliability diagram to %s", reliability_path)


def run_finalize(
    *,
    skip_tune: bool = False,
    params_path: Path = FINAL_PARAMS_JSON,
    n_trials: int = 40,
    n_splits: int = DEFAULT_CV_FOLDS,
    metrics_path: Path = METRICS_CSV,
    trials_path: Path = TUNING_CSV,
    run_shap: bool = False,
    enforce_monotonic: bool = True,
) -> dict:
    """Execute tune (optional), final CV, and calibration artifact generation.

    Args:
        skip_tune: When True, load existing ``params_path`` instead of tuning.
        params_path: Final model params JSON path.
        n_trials: Optuna trials when tuning.
        n_splits: CV folds.
        metrics_path: Output metrics CSV path.
        trials_path: Output tuning trials CSV path.
        run_shap: When True, generate SHAP plots under ``results/figures/``.
        enforce_monotonic: Apply quantile ordering in predictions.

    Returns:
        Final model configuration dict (including ``cv_summary``).
    """
    model_df, feature_cols = build_modeling_table()

    if skip_tune:
        config = load_final_model_params(params_path)
        logger.info("loaded model params from %s", params_path)
    else:
        tuning = tune_lightgbm_quantile_cv(
            model_df,
            feature_cols,
            n_trials=n_trials,
            n_splits=n_splits,
            enforce_monotonic=enforce_monotonic,
        )
        save_tuning_trials(tuning.trials, trials_path)
        config = default_final_config(tuning.best_params)
        config["tuning"] = {
            "best_mean_pinball": tuning.best_score,
            "n_trials": n_trials,
        }

    model_params = dict(config.get("model_params", {}))
    backend = config.get("backend", "lightgbm")
    cat_encoding: CatEncoding = config.get("cat_encoding", "target")
    cat_col = config.get("cat_col", HOUR_SEASON_COL)

    predict_fold = make_quantile_predict_fold(
        feature_cols,
        backend=backend,
        model_params=model_params,
        multi_quantile=config.get("multi_quantile", False),
        cat_col=cat_col,
        cat_encoding=cat_encoding,
        enforce_monotonic=enforce_monotonic,
    )
    result = evaluate_quantile_origin_cv(
        model_df,
        feature_cols,
        predict_fold,
        n_splits=n_splits,
        metrics_path=metrics_path,
    )

    cv_summary = _cv_summary(result.fold_metrics)
    config["cv_summary"] = cv_summary
    save_final_model_params(config, path=params_path, cv_summary=cv_summary)

    if result.oof_predictions is not None and not result.oof_predictions.empty:
        run_calibration_artifacts(result.oof_predictions)
        interval = pi_coverage(
            result.oof_predictions["wind_mw"],
            result.oof_predictions["pred_q10"],
            result.oof_predictions["pred_q90"],
        )
        gap = interval - PI_COVERAGE_TARGET
        direction = "too narrow" if gap < 0 else "too wide" if gap > 0 else "calibrated"
        logger.info(
            "OOF PI coverage=%.4f (target %.2f) — intervals %s",
            interval,
            PI_COVERAGE_TARGET,
            direction,
        )

    if run_shap:
        _run_shap_plots(model_df, feature_cols, config, enforce_monotonic)

    return config


def _run_shap_plots(
    model_df: pd.DataFrame,
    feature_cols: list[str],
    config: dict,
    enforce_monotonic: bool,
) -> None:
    """Fit final model on full sample and write SHAP plots to results/figures/."""
    from wind_quantile_forecast.config import TARGET_COL
    from wind_quantile_forecast.features.target_encoding import (
        add_target_encoding,
        target_encoding_column,
    )
    from wind_quantile_forecast.interpret.shap_explain import explain_model
    from wind_quantile_forecast.models.quantile_gbm import QuantileGBM

    backend = config.get("backend", "lightgbm")
    model_params = dict(config.get("model_params", {}))
    if backend == "lightgbm":
        model_params.setdefault("verbosity", -1)

    te_col = target_encoding_column(HOUR_SEASON_COL)
    encoded_df = add_target_encoding(model_df, HOUR_SEASON_COL, TARGET_COL)
    model_cols = [te_col if c == HOUR_SEASON_COL else c for c in feature_cols]

    model = QuantileGBM(
        backend=backend,
        model_params=model_params,
        multi_quantile=config.get("multi_quantile", False),
    )
    model.fit(encoded_df[model_cols], encoded_df[TARGET_COL])
    RESULTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for q in (0.5, 0.9):
        explain_model(model, encoded_df[model_cols], quantile=q, output_dir=RESULTS_FIGURES_DIR)
    _ = enforce_monotonic


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for finalize pipeline."""
    parser = argparse.ArgumentParser(
        description="Tune, CV, and write calibration artifacts to results/",
    )
    parser.add_argument("--skip-tune", action="store_true", help="Load existing params JSON")
    parser.add_argument("--params-path", type=str, default=str(FINAL_PARAMS_JSON))
    parser.add_argument("--n-trials", type=int, default=40)
    parser.add_argument("--n-splits", type=int, default=DEFAULT_CV_FOLDS)
    parser.add_argument("--metrics-path", type=str, default=str(METRICS_CSV))
    parser.add_argument("--trials-path", type=str, default=str(TUNING_CSV))
    parser.add_argument(
        "--shap",
        action="store_true",
        help="Also write SHAP plots to results/figures/",
    )
    parser.add_argument("--no-monotonic", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    config = run_finalize(
        skip_tune=args.skip_tune,
        params_path=Path(args.params_path),
        n_trials=args.n_trials,
        n_splits=args.n_splits,
        metrics_path=Path(args.metrics_path),
        trials_path=Path(args.trials_path),
        run_shap=args.shap,
        enforce_monotonic=not args.no_monotonic,
    )
    summary = config.get("cv_summary", {})
    print(f"Final mean pinball: {summary.get('mean_pinball', float('nan')):.4f}")
    print(f"PI coverage: {summary.get('pi_coverage', float('nan')):.4f}")
    print(f"Wrote {args.metrics_path}")
    print(f"Wrote {CALIBRATION_CSV}")
    print(f"Wrote {RELIABILITY_PNG}")
    print(f"Wrote {args.params_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
