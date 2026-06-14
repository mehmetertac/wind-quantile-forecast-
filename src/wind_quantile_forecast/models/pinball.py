"""Pinball (quantile) loss for probabilistic forecast evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def pinball_loss(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    quantile: float,
) -> float:
    """Compute mean pinball loss for a single quantile.

    L_q(y, y_hat) = q * max(y - y_hat, 0) + (1 - q) * max(y_hat - y, 0)

    Args:
        y_true: Observed values.
        y_pred: Predicted quantile values.
        quantile: Target quantile in (0, 1).

    Returns:
        Mean pinball loss across all observations.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    residual = yt - yp
    return float(np.mean(np.maximum(quantile * residual, (quantile - 1) * residual)))


__all__ = ["pinball_loss"]
