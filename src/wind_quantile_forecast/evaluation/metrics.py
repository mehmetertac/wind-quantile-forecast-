"""Aggregate evaluation metrics for quantile forecasts."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from wind_quantile_forecast.config import METRICS_CSV, QUANTILES
from wind_quantile_forecast.models.pinball import pinball_loss

logger = logging.getLogger(__name__)

PI_COVERAGE_TARGET = 0.80

METRICS_TABLE_COLUMNS: tuple[str, ...] = (
    "fold",
    "n_train",
    "n_test",
    "pinball_q10",
    "pinball_q50",
    "pinball_q90",
    "pi_coverage",
    "mae",
    "rmse",
    "mape",
)


def pi_coverage(
    y_true: np.ndarray | pd.Series,
    p_lo: np.ndarray | pd.Series,
    p_hi: np.ndarray | pd.Series,
) -> float:
    """Fraction of observations inside the prediction interval [P10, P90].

    For well-calibrated 10 % / 90 % quantiles the nominal coverage is 80 %.

    Args:
        y_true: Observed values.
        p_lo: Lower interval bound (P10).
        p_hi: Upper interval bound (P90).

    Returns:
        Fraction of ``y_true`` values with ``p_lo <= y <= p_hi``.
    """
    yt = np.asarray(y_true, dtype=float)
    lo = np.asarray(p_lo, dtype=float)
    hi = np.asarray(p_hi, dtype=float)
    return float(np.mean((yt >= lo) & (yt <= hi)))


def diagnose_interval_width(
    y_true: np.ndarray | pd.Series,
    p_lo: np.ndarray | pd.Series,
    p_hi: np.ndarray | pd.Series,
    *,
    nominal: float = PI_COVERAGE_TARGET,
) -> dict[str, float]:
    """Diagnose prediction-interval calibration and width.

    Args:
        y_true: Observed values.
        p_lo: Lower interval bound (P10).
        p_hi: Upper interval bound (P90).
        nominal: Nominal PI coverage (default 80%).

    Returns:
        Dict with ``pi_coverage``, ``nominal_coverage``, ``coverage_gap``,
        and ``mean_interval_width``.
    """
    yt = np.asarray(y_true, dtype=float)
    lo = np.asarray(p_lo, dtype=float)
    hi = np.asarray(p_hi, dtype=float)
    observed = pi_coverage(yt, lo, hi)
    return {
        "pi_coverage": observed,
        "nominal_coverage": float(nominal),
        "coverage_gap": observed - float(nominal),
        "mean_interval_width": float(np.mean(hi - lo)),
    }


def mae(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Mean absolute error."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(yt - yp)))


def rmse(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Root mean squared error."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mape(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Mean absolute percentage error (percent).

    Rows where ``y_true == 0`` are excluded. Returns ``nan`` when no non-zero
    observations remain.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = yt != 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100.0)


def evaluate_quantile_forecast(
    y_true: np.ndarray | pd.Series,
    predictions: dict[float, np.ndarray | pd.Series],
    *,
    quantiles: Sequence[float] = QUANTILES,
) -> dict[str, float]:
    """Compute pinball loss, PI coverage, and P50 point-forecast errors.

    Args:
        y_true: Observed wind generation values.
        predictions: Dict mapping quantile -> predicted array (e.g. 0.1, 0.5, 0.9).
        quantiles: Quantile levels used for pinball loss and interval bounds.

    Returns:
        Metrics dict with keys ``pinball_q10``/``q50``/``q90``, ``pi_coverage``,
        ``mae``/``rmse``/``mape`` (on P50), and ``n_test``.
    """
    yt = np.asarray(y_true, dtype=float)
    q_lo, q_med, q_hi = quantiles
    y_med = np.asarray(predictions[q_med], dtype=float)

    metrics: dict[str, float] = {
        "n_test": float(len(yt)),
        "mae": mae(yt, y_med),
        "rmse": rmse(yt, y_med),
        "mape": mape(yt, y_med),
    }
    for q in quantiles:
        yq = np.asarray(predictions[q], dtype=float)
        metrics[f"pinball_q{int(q * 100):02d}"] = pinball_loss(yt, yq, q)
    if q_lo in predictions and q_hi in predictions:
        lo = np.asarray(predictions[q_lo], dtype=float)
        hi = np.asarray(predictions[q_hi], dtype=float)
        metrics["pi_coverage"] = pi_coverage(yt, lo, hi)
    return metrics


def log_fold_metrics(
    fold: int,
    metrics: dict[str, float],
    *,
    log: logging.Logger | None = None,
    pi_target: float = PI_COVERAGE_TARGET,
) -> None:
    """Log per-fold quantile evaluation metrics at INFO level.

    Args:
        fold: Fold index (1-based).
        metrics: Output of :func:`evaluate_quantile_forecast`.
        log: Logger instance; defaults to this module's logger.
        pi_target: Nominal PI coverage for the log message (default 80 %).
    """
    sink = log or logger
    n_test = metrics.get("n_test", float("nan"))
    pinball = " ".join(
        f"{k}={metrics[k]:.4f}"
        for k in sorted(metrics)
        if k.startswith("pinball_") and k in metrics
    )
    point = " ".join(
        f"{name}={metrics[name]:.4f}"
        for name in ("mae", "rmse", "mape")
        if name in metrics and np.isfinite(metrics[name])
    )
    pi_msg = ""
    if "pi_coverage" in metrics:
        pi_msg = (
            f"pi_coverage={metrics['pi_coverage']:.4f} "
            f"(target {pi_target:.2f})"
        )
    sink.info(
        "fold=%d n_test=%.0f | %s | %s | %s",
        fold,
        n_test,
        pinball or "pinball=n/a",
        pi_msg or "pi_coverage=n/a",
        point or "point=n/a",
    )


def format_metrics_table(
    fold_metrics: pd.DataFrame,
    *,
    include_mean: bool = True,
) -> pd.DataFrame:
    """Order rolling-origin CV metrics for tabular export.

    Args:
        fold_metrics: Per-fold metrics from rolling-origin CV
            (:class:`~wind_quantile_forecast.evaluation.cv.RollingOriginCVResult`).
        include_mean: When True, append a summary row (``fold=0``) with column means.

    Returns:
        DataFrame with stable column order ready for CSV export.
    """
    table = fold_metrics.copy()
    for col in METRICS_TABLE_COLUMNS:
        if col not in table.columns:
            table[col] = np.nan
    table = table[list(METRICS_TABLE_COLUMNS)]

    if include_mean and not table.empty:
        numeric = table.select_dtypes(include="number")
        mean_row = numeric.mean(numeric_only=True).to_dict()
        mean_row["fold"] = 0
        if "n_train" in mean_row:
            mean_row["n_train"] = np.nan
        table = pd.concat([table, pd.DataFrame([mean_row])], ignore_index=True)
    return table


def save_metrics_table(
    fold_metrics: pd.DataFrame,
    path: Path | str | None = None,
    *,
    include_mean: bool = True,
) -> Path:
    """Write per-fold CV metrics to ``results/metrics.csv``.

    Args:
        fold_metrics: Per-fold metrics DataFrame from rolling-origin CV.
        path: Output CSV path; defaults to ``config.METRICS_CSV``.
        include_mean: Append a mean summary row when True.

    Returns:
        Resolved path of the written CSV file.
    """
    out_path = Path(path or METRICS_CSV)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = format_metrics_table(fold_metrics, include_mean=include_mean)
    table.to_csv(out_path, index=False, float_format="%.6f")
    logger.info("saved metrics table to %s (%d rows)", out_path, len(table))
    return out_path
