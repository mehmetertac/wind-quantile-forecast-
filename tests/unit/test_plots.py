"""Unit tests for evaluation plots."""

from __future__ import annotations

import numpy as np
from wind_quantile_forecast.evaluation.plots import plot_reliability_diagram


def test_plot_reliability_diagram_writes_png(tmp_path) -> None:
    nominal = np.array([0.1, 0.5, 0.9])
    observed = np.array([0.08, 0.48, 0.88])
    out = tmp_path / "reliability.png"
    plot_reliability_diagram(nominal, observed, out, pi_coverage=0.58)
    assert out.exists()
    assert out.stat().st_size > 0
