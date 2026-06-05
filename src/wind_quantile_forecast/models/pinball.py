"""Pinball (quantile) loss for probabilistic forecast evaluation."""

import numpy as np


def pinball_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
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
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residual = y_true - y_pred
    return float(np.mean(np.maximum(quantile * residual, (quantile - 1) * residual)))
