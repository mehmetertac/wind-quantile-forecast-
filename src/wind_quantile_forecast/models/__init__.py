"""Quantile regression models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from wind_quantile_forecast.models.pinball import pinball_loss
from wind_quantile_forecast.models.quantile_order import (
    count_quantile_crossings,
    enforce_quantile_order,
)

if TYPE_CHECKING:
    from wind_quantile_forecast.models.quantile_gbm import QuantileGBM

_LAZY_EXPORTS = {
    "QuantileGBM": "wind_quantile_forecast.models.quantile_gbm",
    "make_quantile_predict_fold": "wind_quantile_forecast.models.quantile_gbm",
}

__all__ = [
    "QuantileGBM",
    "count_quantile_crossings",
    "enforce_quantile_order",
    "make_quantile_predict_fold",
    "pinball_loss",
]


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    import importlib

    module = importlib.import_module(_LAZY_EXPORTS[name])
    return getattr(module, name)
