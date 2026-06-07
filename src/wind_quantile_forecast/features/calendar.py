"""Calendar and cyclical time features."""

from __future__ import annotations

import pandas as pd
from energy_features.generation_lift import calendar_feature_cols


def calendar_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return calendar and holiday columns present in *df*.

    Calendar features are produced by ``energy_features.calendar.make_calendar_features``
    during dataset build; this helper selects the modeling subset.
    """
    return calendar_feature_cols(df)


def add_calendar_features(df: pd.DataFrame, datetime_col: str = "valid_time") -> pd.DataFrame:
    """No-op passthrough when calendar columns are already joined at ``valid_time``.

    Day-ahead rows carry calendar/holiday fields from the sibling feature pipeline.
    Use :func:`calendar_feature_columns` to list them for modeling.
    """
    if datetime_col not in df.columns:
        msg = f"datetime column {datetime_col!r} not in dataframe"
        raise KeyError(msg)
    return df
