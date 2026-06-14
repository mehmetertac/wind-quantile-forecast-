"""Hyperparameter tuning for quantile GBM models."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd

from wind_quantile_forecast.config import (
    DEFAULT_CV_FOLDS,
    FINAL_PARAMS_JSON,
    HOUR_SEASON_COL,
    QUANTILES,
    RANDOM_SEED,
    TUNING_CSV,
)
from wind_quantile_forecast.evaluation.cv import evaluate_quantile_origin_cv
from wind_quantile_forecast.models import make_quantile_predict_fold

logger = logging.getLogger(__name__)

PINBALL_COLS = ("pinball_q10", "pinball_q50", "pinball_q90")


@dataclass
class TuningResult:
    """Result of an Optuna LightGBM tuning study."""

    best_params: dict[str, Any]
    best_score: float
    trials: pd.DataFrame
    study: optuna.Study


def mean_pinball_cv_score(fold_metrics: pd.DataFrame) -> float:
    """Mean pinball loss across quantiles and CV folds.

    Args:
        fold_metrics: Per-fold metrics from rolling-origin CV (may include fold=0 mean).

    Returns:
        Mean of pinball_q10, pinball_q50, pinball_q90 over real folds only.
    """
    if "fold" in fold_metrics.columns:
        folds = fold_metrics[fold_metrics["fold"] > 0]
    else:
        folds = fold_metrics
    if folds.empty:
        msg = "no fold metrics available for pinball score"
        raise ValueError(msg)
    values: list[float] = []
    for col in PINBALL_COLS:
        if col in folds.columns:
            values.extend(folds[col].dropna().tolist())
    if not values:
        msg = "pinball columns missing from fold metrics"
        raise ValueError(msg)
    return float(np.mean(values))


def _suggest_lightgbm_params(trial: optuna.Trial) -> dict[str, Any]:
    """Sample LightGBM hyperparameters for one Optuna trial."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "verbosity": -1,
    }


def tune_lightgbm_quantile_cv(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    n_trials: int = 40,
    n_splits: int = DEFAULT_CV_FOLDS,
    random_seed: int = RANDOM_SEED,
    enforce_monotonic: bool = True,
    study_name: str = "lightgbm_quantile_pinball",
) -> TuningResult:
    """Tune LightGBM quantile models minimizing mean pinball loss under rolling-origin CV.

    Args:
        df: Modeling table with features and target.
        feature_cols: Feature column names.
        n_trials: Number of Optuna trials.
        n_splits: Rolling-origin CV folds.
        random_seed: Random seed for reproducibility.
        enforce_monotonic: Apply quantile ordering post-processing in each fold.
        study_name: Optuna study identifier.

    Returns:
        :class:`TuningResult` with best params, score, and trial history.
    """
    trial_rows: list[dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_lightgbm_params(trial)
        predict_fold = make_quantile_predict_fold(
            feature_cols,
            backend="lightgbm",
            model_params=params,
            multi_quantile=False,
            cat_col=HOUR_SEASON_COL,
            cat_encoding="target",
            random_seed=random_seed,
            enforce_monotonic=enforce_monotonic,
        )
        result = evaluate_quantile_origin_cv(
            df,
            feature_cols,
            predict_fold,
            n_splits=n_splits,
            metrics_path=None,
        )
        score = mean_pinball_cv_score(result.fold_metrics)
        pi_folds = result.fold_metrics.loc[result.fold_metrics["fold"] > 0, "pi_coverage"]
        pi_cov = float(pi_folds.mean())
        trial_rows.append(
            {
                "trial": trial.number,
                "mean_pinball": score,
                "pi_coverage": pi_cov,
                **params,
            }
        )
        trial.set_user_attr("pi_coverage", pi_cov)
        return score

    sampler = optuna.samplers.TPESampler(seed=random_seed)
    study = optuna.create_study(direction="minimize", study_name=study_name, sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = dict(study.best_params)
    best["verbosity"] = -1
    trials_df = pd.DataFrame(trial_rows)
    return TuningResult(
        best_params=best,
        best_score=float(study.best_value),
        trials=trials_df,
        study=study,
    )


def save_tuning_trials(trials: pd.DataFrame, path: Path | str | None = None) -> Path:
    """Write Optuna trial history to CSV.

    Args:
        trials: Trial metrics DataFrame from :class:`TuningResult`.
        path: Output path; defaults to ``config.TUNING_CSV``.

    Returns:
        Resolved output path.
    """
    out = Path(path or TUNING_CSV)
    out.parent.mkdir(parents=True, exist_ok=True)
    trials.to_csv(out, index=False, float_format="%.6f")
    logger.info("saved tuning trials to %s (%d rows)", out, len(trials))
    return out


def save_final_model_params(
    params: dict[str, Any],
    *,
    path: Path | str | None = None,
    cv_summary: dict[str, float] | None = None,
) -> Path:
    """Persist locked final model configuration as JSON.

    Args:
        params: Model and training configuration dict.
        path: Output path; defaults to ``config.FINAL_PARAMS_JSON``.
        cv_summary: Optional CV metric summary to embed.

    Returns:
        Resolved output path.
    """
    out = Path(path or FINAL_PARAMS_JSON)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **params,
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    if cv_summary is not None:
        payload["cv_summary"] = cv_summary
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("saved final model params to %s", out)
    return out


def load_final_model_params(path: Path | str) -> dict[str, Any]:
    """Load final model params JSON written by :func:`save_final_model_params`.

    Args:
        path: Path to JSON file.

    Returns:
        Parsed configuration dict.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return dict(data)


def default_final_config(best_params: dict[str, Any]) -> dict[str, Any]:
    """Build a standard final-model config dict from tuned LightGBM params.

    Args:
        best_params: Tuned estimator hyperparameters.

    Returns:
        Config dict with backend, encoding, and quantile metadata.
    """
    return {
        "backend": "lightgbm",
        "cat_encoding": "target",
        "cat_col": HOUR_SEASON_COL,
        "quantiles": list(QUANTILES),
        "multi_quantile": False,
        "enforce_monotonic": True,
        "model_params": best_params,
    }
