"""Assemble leakage-safe feature matrix for day-ahead quantile modeling."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from wind_quantile_forecast.config import (
    DAY_AHEAD_LEAD_HOURS,
    HIGH_CARDINALITY_CAT_COLS,
    TARGET_COL,
)
from wind_quantile_forecast.features.calendar import calendar_feature_columns
from wind_quantile_forecast.features.lags import (
    INIT_TIME_COL,
    LEAD_COL,
    VALID_TIME_COL,
    add_lag_features,
)
from wind_quantile_forecast.features.target_encoding import add_hour_season_feature
from wind_quantile_forecast.features.weather import (
    add_weather_driver_features,
    weather_driver_columns,
)

FORBIDDEN_FEATURE_PREFIXES: tuple[str, ...] = ("era5_",)
FORBIDDEN_FEATURE_EXACT: frozenset[str] = frozenset(
    {TARGET_COL, "generation_mw", INIT_TIME_COL, VALID_TIME_COL, "init_time_utc"}
)


def lag_feature_columns(df: pd.DataFrame, *, target_col: str = TARGET_COL) -> list[str]:
    """Lag and rolling columns derived from ``target_col``."""
    prefixes = (f"{target_col}_lag_", f"{target_col}_roll")
    return sorted(c for c in df.columns if c.startswith(prefixes))


def resolve_feature_columns(
    df: pd.DataFrame,
    *,
    target_col: str = TARGET_COL,
    include_lags: bool = True,
    include_high_cardinality: bool = False,
) -> list[str]:
    """Ordered, de-duplicated modeling columns for day-ahead wind quantile GBMs."""
    parts: list[str] = []
    parts.extend(calendar_feature_columns(df))
    parts.extend(weather_driver_columns(df))
    if include_high_cardinality:
        parts.extend(c for c in HIGH_CARDINALITY_CAT_COLS if c in df.columns)
    if include_lags:
        parts.extend(lag_feature_columns(df, target_col=target_col))
    seen: set[str] = set()
    ordered: list[str] = []
    for col in parts:
        if col in seen or col not in df.columns:
            continue
        seen.add(col)
        ordered.append(col)
    return ordered


def validate_no_target_leakage(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    lead_hours: int = DAY_AHEAD_LEAD_HOURS,
    target_col: str = TARGET_COL,
    gen_history: pd.Series | None = None,
    lag_join_key: str = INIT_TIME_COL,
) -> list[str]:
    """Return human-readable violation messages (empty list means OK).

    Checks:
    - No ERA5 hindcast columns (perfect-weather leakage).
    - Target and join metadata excluded from features.
    - ``lead_hours`` constant matches day-ahead horizon when present.
    - Lag features align with ``lag_join_key`` (default ``init_time``), not ``valid_time``.
    """
    violations: list[str] = []

    for col in feature_cols:
        if col in FORBIDDEN_FEATURE_EXACT:
            violations.append(f"{col!r} must not be a model feature")
        for prefix in FORBIDDEN_FEATURE_PREFIXES:
            if col.startswith(prefix):
                violations.append(f"{col!r} is ERA5 hindcast (target leakage risk)")

    if LEAD_COL in df.columns and df[LEAD_COL].nunique() == 1:
        actual_lead = int(df[LEAD_COL].iloc[0])
        if actual_lead != lead_hours:
            violations.append(
                f"lead_hours={actual_lead} does not match expected day-ahead lead={lead_hours}"
            )

    if lag_join_key != INIT_TIME_COL:
        violations.append(
            f"lag_join_key={lag_join_key!r} is unsafe for day-ahead; use {INIT_TIME_COL!r}"
        )

    if gen_history is not None and INIT_TIME_COL in df.columns:
        violations.extend(
            _check_lag_values_at_issue_time(
                df,
                gen_history,
                feature_cols,
                target_col=target_col,
                lag_join_key=lag_join_key,
            )
        )

    return violations


def _check_lag_values_at_issue_time(
    df: pd.DataFrame,
    gen_history: pd.Series,
    feature_cols: Sequence[str],
    *,
    target_col: str,
    lag_join_key: str,
    atol: float = 1e-6,
) -> list[str]:
    """Flag lag columns whose values match a ``valid_time`` join instead of issue time."""
    violations: list[str] = []
    lag_cols = [c for c in feature_cols if c.startswith(f"{target_col}_lag_")]
    if not lag_cols or lag_join_key not in df.columns:
        return violations

    history = gen_history.sort_index().astype(float)
    if history.index.tz is not None and df[lag_join_key].dt.tz is not None:
        if str(history.index.tz) != str(df[lag_join_key].dt.tz):
            history = history.tz_convert(df[lag_join_key].dt.tz)

    for col in lag_cols:
        suffix = col.rsplit("_lag_", maxsplit=1)[-1]
        if not suffix.isdigit():
            continue
        lag_h = int(suffix)
        leak_rows = 0
        for _, row in df.iterrows():
            actual = row[col]
            if pd.isna(actual):
                continue
            issue_t = row[lag_join_key]
            valid_t = row[VALID_TIME_COL]
            lookback_issue = issue_t - pd.Timedelta(hours=lag_h)
            lookback_valid = valid_t - pd.Timedelta(hours=lag_h)
            if lookback_valid <= issue_t:
                continue
            expected_issue = float(history.asof(lookback_issue))
            expected_valid = float(history.asof(lookback_valid))
            matches_issue = abs(actual - expected_issue) <= atol
            matches_valid = abs(actual - expected_valid) <= atol
            if matches_valid and not matches_issue:
                leak_rows += 1
        if leak_rows:
            violations.append(
                f"{col!r} matches valid_time-based lags on {leak_rows} row(s) where "
                f"valid_time - {lag_h}h > init_time; join at {lag_join_key!r} instead"
            )
    return violations


def assemble_feature_matrix(
    df: pd.DataFrame,
    gen_history: pd.Series,
    *,
    lead_hours: int = DAY_AHEAD_LEAD_HOURS,
    target_col: str = TARGET_COL,
    lags: Sequence[int] | None = None,
    windows: Sequence[int] | None = None,
    include_lags: bool = True,
    include_high_cardinality: bool = False,
    validate: bool = True,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Build ``(X, y, feature_cols)`` for day-ahead quantile modeling.

    Pipeline:
    1. Add NWP weather drivers (hub wind, direction sin/cos, air density).
    2. Join causal lag/rolling stats at ``init_time``.
    3. Select calendar + weather + lag columns; assert no target leakage.

    Args:
        df: Day-ahead table from ``prepare_day_ahead_dataset``.
        gen_history: Full hourly wind generation series for autoregressive features.
        lead_hours: Expected forecast horizon (default 24 h).
        target_col: Target column name.
        lags: Optional lag periods; defaults to ``config.DEFAULT_LAGS``.
        windows: Optional rolling windows; defaults to ``config.DEFAULT_ROLL_WINDOWS``.
        include_lags: When False, skip autoregressive block (calendar + weather only).
        include_high_cardinality: Add ``hour_season`` (hour × season, 96 levels).
        validate: Run :func:`validate_no_target_leakage` before returning.

    Returns:
        Feature matrix, target series, and ordered feature column names.
    """
    if LEAD_COL in df.columns:
        at_lead = df.loc[df[LEAD_COL] == lead_hours]
        if at_lead.empty:
            msg = f"no rows at lead_hours={lead_hours}"
            raise ValueError(msg)
        work = at_lead.copy()
    else:
        work = df.copy()

    work = add_weather_driver_features(work)
    if include_high_cardinality:
        work = add_hour_season_feature(work, datetime_col=VALID_TIME_COL)
    if include_lags:
        work = add_lag_features(
            work,
            gen_history,
            target_col=target_col,
            lags=lags,
            windows=windows,
        )

    feature_cols = resolve_feature_columns(
        work,
        target_col=target_col,
        include_lags=include_lags,
        include_high_cardinality=include_high_cardinality,
    )
    if not feature_cols:
        msg = "no modeling features resolved from dataframe"
        raise ValueError(msg)

    if validate:
        violations = validate_no_target_leakage(
            work,
            feature_cols,
            lead_hours=lead_hours,
            target_col=target_col,
            gen_history=gen_history,
            lag_join_key=INIT_TIME_COL,
        )
        if violations:
            lines = "\n".join(f"  - {v}" for v in violations)
            msg = f"feature matrix leakage check failed:\n{lines}"
            raise ValueError(msg)

    if target_col not in work.columns:
        msg = f"target column {target_col!r} not in dataframe"
        raise KeyError(msg)

    X = work[feature_cols].copy()
    y = work[target_col].copy()
    X.attrs["feature_cols"] = feature_cols
    X.attrs["lead_hours"] = lead_hours
    X.attrs["lag_join_key"] = INIT_TIME_COL
    return X, y, feature_cols
