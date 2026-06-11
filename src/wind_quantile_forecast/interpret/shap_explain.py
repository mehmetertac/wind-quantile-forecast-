"""SHAP-based feature importance and explanation plots."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shap

from wind_quantile_forecast.config import QUANTILE_LABELS, RANDOM_SEED, REPORTS_DIR
from wind_quantile_forecast.models.quantile_gbm import QuantileGBM

logger = logging.getLogger(__name__)

DEFAULT_DEPENDENCE_FEATURES: tuple[str, ...] = (
    "nwp_wind_speed_hub_mps",
    "wind_mw_lag_24",
    "wind_mw_roll24_mean",
    "hour_season_te",
)


def _quantile_label(quantile: float) -> str:
    """Return a short label such as ``p50`` for plot filenames."""
    return QUANTILE_LABELS.get(quantile, f"q{quantile:.2f}").lower()


def _sample_frame(X: pd.DataFrame, max_samples: int) -> pd.DataFrame:
    """Subsample rows for SHAP computation when ``X`` is large."""
    if len(X) <= max_samples:
        return X.copy()
    return X.sample(n=max_samples, random_state=RANDOM_SEED)


def explain_model(
    model: QuantileGBM,
    X: pd.DataFrame,
    *,
    quantile: float = 0.5,
    dependence_features: Sequence[str] | None = None,
    output_dir: Path | None = None,
    max_samples: int = 500,
) -> Path:
    """Generate SHAP summary and dependence plots for one quantile model.

    Args:
        model: Fitted :class:`QuantileGBM` with per-quantile estimators.
        X: Feature matrix used for explanation (same columns as fit time).
        quantile: Quantile level to explain (default P50).
        dependence_features: Feature names for dependence plots; defaults to
            wind speed, lag-24, roll-24, and ``hour_season_te``.
        output_dir: Directory to save PNG plots. Defaults to ``reports/figures/``.
        max_samples: Maximum rows to explain (subsampled with ``RANDOM_SEED``).

    Returns:
        The output directory where figures were written.

    Raises:
        ValueError: ``dependence_features`` is empty after filtering missing columns.
    """
    out_dir = output_dir or REPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    label = _quantile_label(quantile)
    estimator = model.estimator_at(quantile)
    X_sample = _sample_frame(X, max_samples)

    explainer = shap.TreeExplainer(estimator, feature_perturbation="tree_path_dependent")
    shap_values = explainer.shap_values(X_sample)

    summary_path = out_dir / f"shap_summary_{label}.png"
    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Wrote %s", summary_path)

    deps = list(dependence_features or DEFAULT_DEPENDENCE_FEATURES)
    present = [f for f in deps if f in X_sample.columns]
    missing = [f for f in deps if f not in X_sample.columns]
    for feat in missing:
        logger.warning("Skipping dependence plot: feature %r not in X", feat)
    if not present:
        msg = f"no dependence features found in X columns: {deps}"
        raise ValueError(msg)

    for feat in present:
        dep_path = out_dir / f"shap_dependence_{label}_{feat}.png"
        plt.figure()
        shap.dependence_plot(feat, shap_values, X_sample, show=False)
        plt.tight_layout()
        plt.savefig(dep_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Wrote %s", dep_path)

    return out_dir
