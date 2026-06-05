"""Lag and rolling window features for autoregressive signals."""

import pandas as pd


def add_lag_features(
    df: pd.DataFrame,
    target_col: str = "wind_mw",
    lags: list[int] | None = None,
) -> pd.DataFrame:
    """Add lagged and rolling-mean features of the target variable.

    Args:
        df: Input DataFrame sorted by time.
        target_col: Name of the wind generation column.
        lags: List of lag periods in hours. Defaults to [1, 6, 12, 24, 48].

    Returns:
        DataFrame with lag features appended.

    Raises:
        NotImplementedError: Feature logic not yet implemented.
    """
    raise NotImplementedError("Lag features not yet implemented")
