"""Unit tests for quantile ordering post-processing."""

from __future__ import annotations

import numpy as np
from wind_quantile_forecast.config import QUANTILES
from wind_quantile_forecast.models.quantile_order import (
    count_quantile_crossings,
    enforce_quantile_order,
)


def test_count_quantile_crossings_detects_inverted_rows() -> None:
    preds = {
        0.1: np.array([10.0, 5.0]),
        0.5: np.array([8.0, 6.0]),
        0.9: np.array([12.0, 4.0]),
    }
    assert count_quantile_crossings(preds, QUANTILES) == 2


def test_enforce_quantile_order_sort_fixes_crossings() -> None:
    preds = {
        0.1: np.array([10.0, 5.0]),
        0.5: np.array([8.0, 6.0]),
        0.9: np.array([12.0, 4.0]),
    }
    ordered = enforce_quantile_order(preds, QUANTILES, method="sort")
    stacked = np.column_stack([ordered[q] for q in sorted(QUANTILES)])
    assert np.all(np.diff(stacked, axis=1) >= 0)
    assert count_quantile_crossings(ordered, QUANTILES) == 0


def test_enforce_quantile_order_isotonic_produces_monotonic_rows() -> None:
    preds = {
        0.1: np.array([10.0, 1.0, 7.0]),
        0.5: np.array([8.0, 9.0, 5.0]),
        0.9: np.array([12.0, 3.0, 11.0]),
    }
    iso_out = enforce_quantile_order(preds, QUANTILES, method="isotonic")
    stacked = np.column_stack([iso_out[q] for q in sorted(QUANTILES)])
    assert np.all(np.diff(stacked, axis=1) >= 0)
    assert count_quantile_crossings(iso_out, QUANTILES) == 0


def test_enforce_quantile_order_already_ordered_unchanged() -> None:
    preds = {
        0.1: np.array([1.0, 2.0]),
        0.5: np.array([5.0, 6.0]),
        0.9: np.array([9.0, 10.0]),
    }
    ordered = enforce_quantile_order(preds, QUANTILES)
    for q in QUANTILES:
        assert np.allclose(ordered[q], preds[q])
