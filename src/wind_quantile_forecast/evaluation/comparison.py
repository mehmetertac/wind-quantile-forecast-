"""Compare LightGBM, XGBoost, and CatBoost on probabilistic CV metrics."""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from wind_quantile_forecast.config import COMPARISON_CSV, HOUR_SEASON_COL, TARGET_COL
from wind_quantile_forecast.evaluation.cv import RollingOriginCVResult, evaluate_quantile_origin_cv
from wind_quantile_forecast.features.target_encoding import CatEncoding
from wind_quantile_forecast.models.quantile_gbm import Backend, make_quantile_predict_fold

logger = logging.getLogger(__name__)

COMPARISON_METRIC_COLUMNS: tuple[str, ...] = (
    "pinball_q10",
    "pinball_q50",
    "pinball_q90",
    "pi_coverage",
    "mae",
    "rmse",
    "mape",
)

COMPARISON_TABLE_COLUMNS: tuple[str, ...] = (
    "backend",
    "encoding",
    "pinball_q10",
    "pinball_q50",
    "pinball_q90",
    "pi_coverage",
    "mae",
    "rmse",
    "mape",
    "train_time_sec",
    "n_folds",
)


@dataclass(frozen=True)
class BackendComparisonSpec:
    """One row in the backend comparison table."""

    backend: Backend
    encoding: CatEncoding
    label: str | None = None

    @property
    def row_label(self) -> str:
        if self.label:
            return self.label
        if self.encoding == "none":
            return self.backend
        return f"{self.backend}+{self.encoding}"


DEFAULT_BACKEND_SPECS: tuple[BackendComparisonSpec, ...] = (
    BackendComparisonSpec("lightgbm", "target"),
    BackendComparisonSpec("xgboost", "target"),
    BackendComparisonSpec("catboost", "target"),
    BackendComparisonSpec("catboost", "native", label="catboost+native_cat"),
)


def summarize_fold_metrics(fold_metrics: pd.DataFrame) -> dict[str, float]:
    """Mean probabilistic metrics across CV folds."""
    summary: dict[str, float] = {}
    for col in COMPARISON_METRIC_COLUMNS:
        if col in fold_metrics.columns:
            summary[col] = float(fold_metrics[col].mean())
    return summary


def run_backend_cv(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    spec: BackendComparisonSpec,
    *,
    cat_col: str | None = HOUR_SEASON_COL,
    target_col: str = TARGET_COL,
    n_splits: int = 5,
    model_params: dict | None = None,
    multi_quantile: bool = True,
) -> tuple[RollingOriginCVResult, float]:
    """Run rolling-origin CV for one backend/encoding configuration.

    Args:
        df: Modeling table with features, target, and ``valid_time``.
        feature_cols: Feature column names.
        spec: Backend and encoding mode.
        cat_col: High-cardinality categorical column encoded inside folds.
        target_col: Target column.
        n_splits: Number of CV folds.
        model_params: Estimator kwargs passed to each fold.
        multi_quantile: Use multi-quantile training for XGBoost/CatBoost.

    Returns:
        ``(cv_result, train_time_sec)`` where training time covers all folds.
    """
    encoding: CatEncoding = spec.encoding if cat_col else "none"
    predict_fold = make_quantile_predict_fold(
        feature_cols,
        target_col=target_col,
        backend=spec.backend,
        model_params=model_params,
        multi_quantile=multi_quantile,
        cat_col=cat_col,
        cat_encoding=encoding,
    )
    started = time.perf_counter()
    result = evaluate_quantile_origin_cv(
        df,
        feature_cols,
        predict_fold,
        target_col=target_col,
        n_splits=n_splits,
        metrics_path=None,
    )
    train_time_sec = time.perf_counter() - started
    return result, train_time_sec


def summarize_backend_result(
    spec: BackendComparisonSpec,
    result: RollingOriginCVResult,
    train_time_sec: float,
) -> dict[str, float | str | int]:
    """Build one comparison-table row from a CV result."""
    metrics = summarize_fold_metrics(result.fold_metrics)
    row: dict[str, float | str | int] = {
        "backend": spec.backend,
        "encoding": spec.encoding,
        "train_time_sec": float(train_time_sec),
        "n_folds": int(len(result.fold_metrics)),
    }
    row.update(metrics)
    return row


def format_comparison_table(rows: Sequence[dict[str, float | str | int]]) -> pd.DataFrame:
    """Order backend comparison rows for CSV export."""
    table = pd.DataFrame(rows)
    for col in COMPARISON_TABLE_COLUMNS:
        if col not in table.columns:
            table[col] = np.nan
    return table[list(COMPARISON_TABLE_COLUMNS)]


def save_comparison_table(
    rows: Sequence[dict[str, float | str | int]],
    path: Path | str | None = None,
) -> Path:
    """Write backend comparison table to CSV.

    Args:
        rows: Output of :func:`summarize_backend_result` for each spec.
        path: Output path; defaults to ``config.COMPARISON_CSV``.

    Returns:
        Resolved path of the written CSV.
    """
    out_path = Path(path or COMPARISON_CSV)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = format_comparison_table(rows)
    table.to_csv(out_path, index=False, float_format="%.6f")
    logger.info("saved backend comparison to %s (%d rows)", out_path, len(table))
    return out_path


def run_backend_comparison(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    specs: Sequence[BackendComparisonSpec] = DEFAULT_BACKEND_SPECS,
    cat_col: str | None = HOUR_SEASON_COL,
    target_col: str = TARGET_COL,
    n_splits: int = 5,
    model_params: dict | None = None,
    multi_quantile: bool = True,
    comparison_path: Path | str | None = COMPARISON_CSV,
) -> pd.DataFrame:
    """Run CV for each backend and return a comparison table.

    LightGBM and XGBoost use fold-wise target encoding on ``cat_col``.
    CatBoost is evaluated with both target encoding and native categoricals.

    Args:
        df: Modeling table.
        feature_cols: Feature columns (should include ``cat_col`` when set).
        specs: Backend/encoding configurations to compare.
        cat_col: High-cardinality categorical feature.
        target_col: Target column.
        n_splits: CV folds per backend.
        model_params: Shared estimator kwargs.
        multi_quantile: Multi-quantile mode for XGBoost/CatBoost.
        comparison_path: CSV output path; pass ``None`` to skip writing.

    Returns:
        Comparison DataFrame with probabilistic metrics and ``train_time_sec``.
    """
    rows: list[dict[str, float | str | int]] = []
    for spec in specs:
        logger.info("comparing backend=%s encoding=%s", spec.backend, spec.encoding)
        result, train_time = run_backend_cv(
            df,
            feature_cols,
            spec,
            cat_col=cat_col,
            target_col=target_col,
            n_splits=n_splits,
            model_params=model_params,
            multi_quantile=multi_quantile,
        )
        rows.append(summarize_backend_result(spec, result, train_time))

    table = format_comparison_table(rows)
    if comparison_path is not None:
        save_comparison_table(rows, comparison_path)
    return table


def comparison_table_markdown(table: pd.DataFrame) -> str:
    """Render the comparison table as a GitHub-flavored markdown table."""
    display = table.copy()
    for col in display.columns:
        if col in {"backend", "encoding"}:
            continue
        if col == "n_folds":
            display[col] = display[col].astype(int)
            continue
        display[col] = display[col].map(lambda v: f"{v:.4f}" if pd.notna(v) else "n/a")
    headers = "| " + " | ".join(display.columns) + " |"
    sep = "| " + " | ".join("---" for _ in display.columns) + " |"
    body = "\n".join(
        "| " + " | ".join(str(row[col]) for col in display.columns) + " |"
        for _, row in display.iterrows()
    )
    return "\n".join([headers, sep, body])
