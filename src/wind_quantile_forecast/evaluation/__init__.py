"""Forecast evaluation: pinball loss, coverage, reliability diagrams."""

from wind_quantile_forecast.evaluation.comparison import (
    COMPARISON_TABLE_COLUMNS,
    DEFAULT_BACKEND_SPECS,
    BackendComparisonSpec,
    comparison_table_markdown,
    format_comparison_table,
    run_backend_comparison,
    save_comparison_table,
)
from wind_quantile_forecast.evaluation.cv import (
    RollingOriginCVResult,
    RollingOriginSplit,
    TimeFold,
    default_quantile_fold_metrics,
    evaluate_quantile_origin_cv,
    rolling_origin_time_folds,
    run_rolling_origin_cv,
)
from wind_quantile_forecast.evaluation.metrics import (
    METRICS_TABLE_COLUMNS,
    PI_COVERAGE_TARGET,
    evaluate_quantile_forecast,
    format_metrics_table,
    log_fold_metrics,
    mae,
    mape,
    pi_coverage,
    pinball_loss,
    rmse,
    save_metrics_table,
)
from wind_quantile_forecast.evaluation.plots import plot_reliability_diagram
from wind_quantile_forecast.evaluation.reliability import compute_reliability_curve

__all__ = [
    "COMPARISON_TABLE_COLUMNS",
    "DEFAULT_BACKEND_SPECS",
    "BackendComparisonSpec",
    "METRICS_TABLE_COLUMNS",
    "PI_COVERAGE_TARGET",
    "RollingOriginCVResult",
    "RollingOriginSplit",
    "TimeFold",
    "comparison_table_markdown",
    "compute_reliability_curve",
    "default_quantile_fold_metrics",
    "evaluate_quantile_forecast",
    "evaluate_quantile_origin_cv",
    "format_comparison_table",
    "format_metrics_table",
    "log_fold_metrics",
    "mae",
    "mape",
    "pinball_loss",
    "pi_coverage",
    "plot_reliability_diagram",
    "run_backend_comparison",
    "save_comparison_table",
    "rmse",
    "save_metrics_table",
    "rolling_origin_time_folds",
    "run_rolling_origin_cv",
]
