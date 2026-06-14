"""Reliability (calibration) curve computation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from wind_quantile_forecast.config import QUANTILES


def marginal_quantile_coverage(
    y_true: np.ndarray | Sequence[float],
    predictions: Mapping[float, np.ndarray | Sequence[float]],
) -> dict[float, float]:
    """Compute empirical coverage at each predicted quantile level.

    For a well-calibrated quantile ``q``, ``mean(y <= pred_q)`` should equal ``q``.

    Args:
        y_true: Observed values.
        predictions: Dict mapping quantile level -> predicted array.

    Returns:
        Dict mapping quantile -> empirical coverage frequency.
    """
    yt = np.asarray(y_true, dtype=float)
    return {
        q: float(np.mean(yt <= np.asarray(pred, dtype=float)))
        for q, pred in predictions.items()
    }


def quantile_calibration_gap(
    observed: np.ndarray | Sequence[float],
    nominal: np.ndarray | Sequence[float],
) -> float:
    """Mean absolute deviation between empirical and nominal quantile coverage.

    Args:
        observed: Empirical coverage frequencies.
        nominal: Nominal quantile levels.

    Returns:
        Mean absolute gap (0 = perfect calibration on the reported points).
    """
    obs = np.asarray(observed, dtype=float)
    nom = np.asarray(nominal, dtype=float)
    if obs.shape != nom.shape:
        msg = "observed and nominal arrays must have the same shape"
        raise ValueError(msg)
    return float(np.mean(np.abs(obs - nom)))


def compute_reliability_curve(
    y_true: np.ndarray,
    predictions: dict[float, np.ndarray],
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute observed vs nominal quantile coverage for reliability diagrams.

    Uses marginal coverage at each quantile level present in ``predictions``.
    The ``n_bins`` parameter is reserved for future binned reliability curves
    and does not affect the current per-quantile computation.

    Args:
        y_true: Observed values.
        predictions: Dict mapping quantile -> predicted array.
        n_bins: Reserved for future binned reliability; currently unused.

    Returns:
        Tuple of (nominal_quantiles, observed_frequencies), sorted by quantile.
    """
    _ = n_bins
    if not predictions:
        return np.array([]), np.array([])

    coverage = marginal_quantile_coverage(y_true, predictions)
    nominal = np.array(sorted(coverage.keys()), dtype=float)
    observed = np.array([coverage[q] for q in nominal], dtype=float)
    return nominal, observed


def reliability_from_oof(
    oof: pd.DataFrame,
    *,
    target_col: str = "wind_mw",
    quantiles: Sequence[float] = QUANTILES,
) -> tuple[np.ndarray, np.ndarray]:
    """Build reliability curve arrays from an OOF prediction DataFrame.

    Args:
        oof: Out-of-fold predictions with ``pred_q10``/``pred_q50``/``pred_q90``.
        target_col: Observed target column name.
        quantiles: Quantile levels corresponding to OOF prediction columns.

    Returns:
        Tuple of (nominal_quantiles, observed_frequencies).
    """
    if oof.empty:
        return np.array([]), np.array([])

    predictions = {
        q: oof[f"pred_q{int(q * 100):02d}"].to_numpy(dtype=float) for q in quantiles
    }
    return compute_reliability_curve(
        oof[target_col].to_numpy(dtype=float),
        predictions,
    )


def build_calibration_table(
    oof: pd.DataFrame,
    *,
    target_col: str = "wind_mw",
    quantiles: Sequence[float] = QUANTILES,
    nominal_pi: float = 0.80,
) -> pd.DataFrame:
    """Build per-quantile and interval calibration diagnostics from OOF preds.

    Args:
        oof: Out-of-fold predictions DataFrame.
        target_col: Observed target column.
        quantiles: Quantile levels.
        nominal_pi: Nominal P10-P90 coverage target.

    Returns:
        DataFrame with per-quantile rows plus one interval summary row.
    """
    from wind_quantile_forecast.evaluation.metrics import diagnose_interval_width

    if oof.empty:
        return pd.DataFrame()

    y = oof[target_col].to_numpy(dtype=float)
    predictions = {
        q: oof[f"pred_q{int(q * 100):02d}"].to_numpy(dtype=float) for q in quantiles
    }
    marginal = marginal_quantile_coverage(y, predictions)
    rows: list[dict[str, float | str]] = []
    for q in sorted(quantiles):
        obs = marginal[q]
        rows.append(
            {
                "metric": f"quantile_{int(q * 100):02d}",
                "nominal": q,
                "observed": obs,
                "gap": obs - q,
            }
        )

    q_lo, q_hi = min(quantiles), max(quantiles)
    interval = diagnose_interval_width(
        y,
        predictions[q_lo],
        predictions[q_hi],
        nominal=nominal_pi,
    )
    rows.append(
        {
            "metric": "pi_interval",
            "nominal": interval["nominal_coverage"],
            "observed": interval["pi_coverage"],
            "gap": interval["coverage_gap"],
            "mean_interval_width": interval["mean_interval_width"],
        }
    )
    return pd.DataFrame(rows)
