"""Visualization: reliability diagrams and forecast plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wind_quantile_forecast.config import QUANTILE_LABELS
from wind_quantile_forecast.evaluation.metrics import PI_COVERAGE_TARGET


def plot_reliability_diagram(
    nominal: np.ndarray,
    observed: np.ndarray,
    output_path: Path | None = None,
    *,
    pi_coverage: float | None = None,
    pi_target: float = PI_COVERAGE_TARGET,
) -> None:
    """Plot a reliability (calibration) diagram with optional interval coverage panel.

    Args:
        nominal: Nominal quantile levels (x-axis).
        observed: Observed coverage frequencies (y-axis).
        output_path: If provided, save figure to this path.
        pi_coverage: Optional empirical P10-P90 interval coverage for bar panel.
        pi_target: Nominal interval coverage target (default 80%).
    """
    if len(nominal) == 0:
        msg = "nominal quantile array is empty"
        raise ValueError(msg)

    fig, axes = plt.subplots(1, 2 if pi_coverage is not None else 1, figsize=(10, 4))
    if pi_coverage is not None:
        ax_rel, ax_pi = axes
    else:
        ax_rel = axes if isinstance(axes, plt.Axes) else axes[0]

    ax_rel.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    ax_rel.scatter(nominal, observed, s=60, zorder=3)
    for q, obs in zip(nominal, observed, strict=True):
        label = QUANTILE_LABELS.get(q, f"q={q:.2f}")
        ax_rel.annotate(
            f"{label}\n{obs:.2f}",
            (q, obs),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
        )
    ax_rel.set_xlim(0, 1)
    ax_rel.set_ylim(0, 1)
    ax_rel.set_xlabel("Nominal quantile")
    ax_rel.set_ylabel("Empirical coverage")
    ax_rel.set_title("Reliability diagram")
    ax_rel.set_aspect("equal")
    ax_rel.grid(True, alpha=0.3)
    ax_rel.legend(loc="lower right")

    if pi_coverage is not None:
        ax_pi.bar(
            ["Empirical", "Nominal"],
            [pi_coverage, pi_target],
            color=["steelblue", "lightgray"],
        )
        gap = pi_coverage - pi_target
        direction = "narrow" if gap < 0 else "wide" if gap > 0 else "calibrated"
        ax_pi.set_ylim(0, 1)
        ax_pi.set_ylabel("P10-P90 coverage")
        ax_pi.set_title(f"Interval coverage ({direction})")
        ax_pi.axhline(pi_target, color="k", linestyle="--", linewidth=0.8)

    fig.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
