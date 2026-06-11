"""Shared helpers for building the day-ahead modeling table."""

from __future__ import annotations

import pandas as pd

from wind_quantile_forecast.config import (
    DAY_AHEAD_LEAD_HOURS,
    TARGET_COL,
    VALID_TIME_COL,
)
from wind_quantile_forecast.data.dataset import load_wind_generation
from wind_quantile_forecast.data.preprocess import load_day_ahead_dataset
from wind_quantile_forecast.features import assemble_feature_matrix
from wind_quantile_forecast.features.lags import LEAD_COL


def build_modeling_table(
    *,
    include_high_cardinality: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Load day-ahead data and return a modeling frame with feature columns.

    Args:
        include_high_cardinality: When ``True``, include ``hour_season`` (96 levels).

    Returns:
        ``(model_df, feature_cols)`` where ``model_df`` holds features, target,
        and ``valid_time`` for rolling-origin CV or final-fit scripts.
    """
    day_ahead = load_day_ahead_dataset()
    gen_history = load_wind_generation()
    at_lead = day_ahead.loc[day_ahead[LEAD_COL] == DAY_AHEAD_LEAD_HOURS]
    X, y, feature_cols = assemble_feature_matrix(
        day_ahead,
        gen_history,
        include_high_cardinality=include_high_cardinality,
    )
    model_df = X.copy()
    model_df[TARGET_COL] = y
    model_df[VALID_TIME_COL] = at_lead.loc[X.index, VALID_TIME_COL]
    return model_df, feature_cols
