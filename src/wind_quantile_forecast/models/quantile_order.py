"""Post-processing to enforce monotonic quantile order (P10 <= P50 <= P90)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from sklearn.isotonic import IsotonicRegression

from wind_quantile_forecast.config import QUANTILES

OrderMethod = Literal["sort", "isotonic"]


def count_quantile_crossings(
    predictions: dict[float, np.ndarray],
    quantiles: Sequence[float] = QUANTILES,
) -> int:
    """Count rows where predicted quantiles are not monotonically increasing.

    Args:
        predictions: Dict mapping quantile level -> predicted array.
        quantiles: Ordered quantile levels to check.

    Returns:
        Number of observations with at least one crossing.
    """
    q_list = sorted(quantiles)
    if len(q_list) < 2:
        return 0
    missing = [q for q in q_list if q not in predictions]
    if missing:
        msg = f"predictions missing quantile levels: {missing}"
        raise KeyError(msg)

    stacked = np.column_stack([np.asarray(predictions[q], dtype=float) for q in q_list])
    diffs = np.diff(stacked, axis=1)
    return int(np.sum(np.any(diffs < 0, axis=1)))


def enforce_quantile_order(
    predictions: dict[float, np.ndarray],
    quantiles: Sequence[float] = QUANTILES,
    method: OrderMethod = "sort",
) -> dict[float, np.ndarray]:
    """Ensure P10 <= P50 <= P90 (or general monotonic order) per row.

    Args:
        predictions: Dict mapping quantile level -> predicted array.
        quantiles: Ordered quantile levels to enforce.
        method: ``"sort"`` applies row-wise ``np.sort``; ``"isotonic"`` fits
            increasing isotonic regression on quantile knots per row.

    Returns:
        New dict with monotonically ordered predictions (same keys as input).
    """
    q_list = sorted(quantiles)
    missing = [q for q in q_list if q not in predictions]
    if missing:
        msg = f"predictions missing quantile levels: {missing}"
        raise KeyError(msg)

    arrays = [np.asarray(predictions[q], dtype=float) for q in q_list]
    n = len(arrays[0])
    for arr in arrays[1:]:
        if len(arr) != n:
            msg = "all quantile prediction arrays must have the same length"
            raise ValueError(msg)

    stacked = np.column_stack(arrays)

    if method == "sort":
        ordered = np.sort(stacked, axis=1)
    elif method == "isotonic":
        ordered = np.empty_like(stacked)
        q_arr = np.asarray(q_list, dtype=float)
        for i in range(n):
            iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
            ordered[i] = iso.fit_transform(q_arr, stacked[i])
    else:
        msg = f"unsupported order method: {method!r}"
        raise ValueError(msg)

    return {q: ordered[:, j] for j, q in enumerate(q_list)}
