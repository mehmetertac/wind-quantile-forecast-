"""Calendar and cyclical time features."""

import pandas as pd


def add_calendar_features(df: pd.DataFrame, datetime_col: str = "timestamp") -> pd.DataFrame:
    """Add hour-of-day, day-of-week, month, and cyclical sin/cos encodings.

    Args:
        df: Input DataFrame with a datetime column.
        datetime_col: Name of the datetime column.

    Returns:
        DataFrame with calendar features appended.

    Raises:
        NotImplementedError: Feature logic not yet implemented.
    """
    raise NotImplementedError("Calendar features not yet implemented")
