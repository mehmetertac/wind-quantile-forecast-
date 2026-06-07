"""Lag and rolling window features for autoregressive signals (horizon-safe)."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from energy_features.lags import make_lag_rolling_block

from wind_quantile_forecast.config import (
    DEFAULT_LAGS,
    DEFAULT_ROLL_WINDOWS,
    TARGET_COL,
)

INIT_TIME_COL = "init_time"
VALID_TIME_COL = "valid_time"
LEAD_COL = "lead_hours"


def _require_day_ahead_cols(df: pd.DataFrame) -> None:
    missing = {INIT_TIME_COL, VALID_TIME_COL, LEAD_COL} - set(df.columns)
    if missing:
        msg = f"day-ahead frame missing columns: {sorted(missing)}"
        raise ValueError(msg)


def add_lag_features(
    df: pd.DataFrame,
    gen_history: pd.Series,
    *,
    target_col: str = TARGET_COL,
    lags: Sequence[int] | None = None,
    windows: Sequence[int] | None = None,
    shift: int = 1,
) -> pd.DataFrame:
    """Add lagged and rolling-mean features joined at ``init_time``.

    Autoregressive features are computed on the hourly generation history and merged
    onto each row by ``init_time`` (forecast issue time). Values at or after
    ``valid_time`` therefore never enter the feature vector, which prevents target
    leakage across the day-ahead horizon.

    Args:
        df: Day-ahead modeling frame with ``valid_time``, ``init_time``, ``lead_hours``.
        gen_history: Hourly wind generation indexed by ``DatetimeIndex``.
        target_col: Prefix for lag/rolling column names.
        lags: Lag periods in hours (default ``config.DEFAULT_LAGS``).
        windows: Rolling window lengths in hours (default ``config.DEFAULT_ROLL_WINDOWS``).
        shift: Causal shift for rolling stats (``>= 1`` excludes contemporaneous target).

    Returns:
        Copy of ``df`` with lag and rolling columns appended.
    """
    _require_day_ahead_cols(df)
    lag_list = list(lags if lags is not None else DEFAULT_LAGS)
    win_list = list(windows if windows is not None else DEFAULT_ROLL_WINDOWS)

    history = gen_history.sort_index().astype(float)
    if history.index.tz is not None and df[INIT_TIME_COL].dt.tz is not None:
        if str(history.index.tz) != str(df[INIT_TIME_COL].dt.tz):
            history = history.tz_convert(df[INIT_TIME_COL].dt.tz)

    history = history.rename(target_col)
    lag_block = make_lag_rolling_block(
        history,
        lag_list,
        win_list,
        shift=shift,
        prefix=target_col,
    )

    out = df.copy()
    joined = out[[INIT_TIME_COL]].merge(
        lag_block,
        left_on=INIT_TIME_COL,
        right_index=True,
        how="left",
    )
    for col in lag_block.columns:
        out[col] = joined[col].to_numpy()
    return out
