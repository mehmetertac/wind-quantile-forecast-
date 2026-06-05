"""Visualization: reliability diagrams and forecast plots."""

from pathlib import Path

import numpy as np


def plot_reliability_diagram(
    nominal: np.ndarray,
    observed: np.ndarray,
    output_path: Path | None = None,
) -> None:
    """Plot a reliability (calibration) diagram.

    Args:
        nominal: Nominal quantile levels.
        observed: Observed coverage frequencies.
        output_path: If provided, save figure to this path.

    Raises:
        NotImplementedError: Plotting logic not yet implemented.
    """
    raise NotImplementedError("Reliability diagram plotting not yet implemented")
