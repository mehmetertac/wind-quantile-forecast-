"""Aggregate evaluation metrics for quantile forecasts."""

import numpy as np


def evaluate_quantile_forecast(
    y_true: np.ndarray,
    predictions: dict[float, np.ndarray],
) -> dict[str, float]:
    """Compute pinball loss and coverage for each quantile.

    Args:
        y_true: Observed wind generation values.
        predictions: Dict mapping quantile -> predicted array.

    Returns:
        Dict with keys like ``pinball_q0.1``, ``coverage_q0.1``, etc.

    Raises:
        NotImplementedError: Evaluation logic not yet implemented.
    """
    raise NotImplementedError("Quantile forecast evaluation not yet implemented")
