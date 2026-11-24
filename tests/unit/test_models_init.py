"""Regression tests for models package lazy exports."""

from __future__ import annotations

from wind_quantile_forecast.models import pinball_loss
from wind_quantile_forecast.models.pinball import pinball_loss as pinball_direct


def test_pinball_import_without_quantile_gbm_eager_load() -> None:
    """Importing pinball_loss must not require energy_features (lazy GBM exports)."""
    import numpy as np

    y = np.array([1.0])
    assert pinball_loss(y, y, quantile=0.5) == pinball_direct(y, y, quantile=0.5)
