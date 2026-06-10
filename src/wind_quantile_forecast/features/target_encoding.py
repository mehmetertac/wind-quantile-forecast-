"""Target encoding for high-cardinality categorical features."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from wind_quantile_forecast.config import (
    DEFAULT_TARGET_ENCODING_SMOOTHING,
    HOUR_SEASON_COL,
    TARGET_COL,
    VALID_TIME_COL,
)

CatEncoding = Literal["target", "native", "none"]


def target_encoding_column(cat_col: str) -> str:
    """Return the numeric column name produced by target encoding."""
    return f"{cat_col}_te"


def season_bucket(month: int) -> int:
    """Meteorological season index (0=winter … 3=autumn) for a calendar month."""
    if month in (12, 1, 2):
        return 0
    if month in (3, 4, 5):
        return 1
    if month in (6, 7, 8):
        return 2
    return 3


def add_hour_season_feature(
    df: pd.DataFrame,
    *,
    datetime_col: str = VALID_TIME_COL,
    out_col: str = HOUR_SEASON_COL,
) -> pd.DataFrame:
    """Add hour-of-day × season categorical feature (96 combinations).

    Args:
        df: Input frame with a datetime column.
        datetime_col: Column used to derive hour and season.
        out_col: Output categorical column name.

    Returns:
        Copy of ``df`` with ``out_col`` (string labels like ``"14_2"``).
    """
    if datetime_col not in df.columns:
        msg = f"datetime column {datetime_col!r} not in dataframe"
        raise KeyError(msg)
    out = df.copy()
    dt = pd.to_datetime(out[datetime_col])
    hours = dt.dt.hour
    seasons = dt.dt.month.map(season_bucket)
    out[out_col] = hours.astype(str) + "_" + seasons.astype(str)
    return out


class TargetEncoder:
    """Smoothed target encoder fit on training data only."""

    def __init__(self, smoothing: float = DEFAULT_TARGET_ENCODING_SMOOTHING) -> None:
        self.smoothing = smoothing
        self.global_mean_: float = 0.0
        self.mapping_: dict[str, float] = {}
        self.counts_: dict[str, int] = {}
        self.cat_sum_: dict[str, float] = {}

    def fit(self, categories: pd.Series, target: pd.Series) -> TargetEncoder:
        """Learn category statistics from training targets.

        Args:
            categories: High-cardinality categorical values.
            target: Training target used for encoding.

        Returns:
            Fitted encoder.
        """
        cats = categories.astype(str)
        y = target.astype(float)
        self.global_mean_ = float(y.mean()) if len(y) else 0.0
        self.mapping_.clear()
        self.counts_.clear()
        self.cat_sum_.clear()

        frame = pd.DataFrame({"cat": cats, "y": y})
        for cat, grp in frame.groupby("cat", sort=False)["y"]:
            cat_sum = float(grp.sum())
            count = int(len(grp))
            smoothed = (cat_sum + self.smoothing * self.global_mean_) / (
                count + self.smoothing
            )
            self.cat_sum_[cat] = cat_sum
            self.counts_[cat] = count
            self.mapping_[cat] = smoothed
        return self

    def transform(
        self,
        categories: pd.Series,
        target: pd.Series | None = None,
    ) -> np.ndarray:
        """Encode categories using train-fitted statistics.

        When ``target`` is provided (training fold), uses leave-one-out encoding
        so each row's own target does not influence its encoded value.

        Args:
            categories: Values to encode.
            target: Optional training targets for LOO encoding.

        Returns:
            Encoded numeric array aligned with ``categories``.
        """
        cats = categories.astype(str).to_numpy()
        if target is not None:
            return self._transform_loo(cats, target.astype(float).to_numpy())
        return np.array([self.mapping_.get(str(c), self.global_mean_) for c in cats], dtype=float)

    def _transform_loo(self, cats: np.ndarray, target: np.ndarray) -> np.ndarray:
        result = np.empty(len(cats), dtype=float)
        for i, (cat, yi) in enumerate(zip(cats, target, strict=True)):
            count = self.counts_.get(str(cat), 0)
            if count <= 1:
                result[i] = self.global_mean_
                continue
            cat_sum = self.cat_sum_[str(cat)]
            result[i] = (cat_sum - yi + self.smoothing * self.global_mean_) / (
                count - 1 + self.smoothing
            )
        return result


def encode_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cat_col: str,
    target_col: str = TARGET_COL,
    *,
    smoothing: float = DEFAULT_TARGET_ENCODING_SMOOTHING,
    out_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Target-encode a categorical column inside one CV fold (no leakage).

    Statistics are fit on ``train_df`` only. Training rows use leave-one-out
    encoding; test rows use the train mapping with global-mean fallback for
    unseen categories.

    Args:
        train_df: Training fold.
        test_df: Test fold.
        cat_col: Categorical column to encode.
        target_col: Target column for encoding.
        smoothing: Smoothing factor for rare categories.
        out_col: Output column name (default ``{cat_col}_te``).

    Returns:
        ``(train_df, test_df, out_col)`` with encoded numeric column added.
    """
    encoded_col = out_col or target_encoding_column(cat_col)
    encoder = TargetEncoder(smoothing=smoothing)
    encoder.fit(train_df[cat_col], train_df[target_col])

    train_out = train_df.copy()
    test_out = test_df.copy()
    train_out[encoded_col] = encoder.transform(train_df[cat_col], train_df[target_col])
    test_out[encoded_col] = encoder.transform(test_df[cat_col])
    return train_out, test_out, encoded_col


def apply_categorical_encoding(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cat_col: str,
    target_col: str,
    *,
    encoding: CatEncoding,
    smoothing: float = DEFAULT_TARGET_ENCODING_SMOOTHING,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str] | None]:
    """Prepare train/test frames for a modeling backend.

    Args:
        train_df: Training fold.
        test_df: Test fold.
        cat_col: High-cardinality categorical column.
        target_col: Target column for target encoding.
        encoding: ``"target"`` replaces the column with a numeric encoding;
            ``"native"`` keeps the raw categorical for CatBoost;
            ``"none"`` leaves frames unchanged.
        smoothing: Smoothing for target encoding.

    Returns:
        ``(train_df, test_df, cat_features)`` where ``cat_features`` is a
        one-element list for native CatBoost mode, else ``None``.
    """
    if encoding == "none" or cat_col not in train_df.columns:
        return train_df, test_df, None
    if encoding == "target":
        tr, te, encoded_col = encode_fold(
            train_df,
            test_df,
            cat_col,
            target_col,
            smoothing=smoothing,
        )
        return tr, te, [encoded_col]
    if encoding == "native":
        tr = train_df.copy()
        te = test_df.copy()
        tr[cat_col] = tr[cat_col].astype(str)
        te[cat_col] = te[cat_col].astype(str)
        return tr, te, [cat_col]
    msg = f"unsupported encoding mode: {encoding!r}"
    raise ValueError(msg)


def add_target_encoding(
    df: pd.DataFrame,
    cat_col: str,
    target_col: str = TARGET_COL,
    smoothing: float = DEFAULT_TARGET_ENCODING_SMOOTHING,
) -> pd.DataFrame:
    """Apply in-sample LOO target encoding (for exploratory analysis only).

    For production/CV use :func:`encode_fold` so statistics are fit on the
    training fold only.

    Args:
        df: Input DataFrame.
        cat_col: Categorical column to encode.
        target_col: Target column used for encoding.
        smoothing: Smoothing factor for rare categories.

    Returns:
        DataFrame with ``{cat_col}_te`` added.
    """
    encoder = TargetEncoder(smoothing=smoothing)
    encoder.fit(df[cat_col], df[target_col])
    out = df.copy()
    out_col = target_encoding_column(cat_col)
    out[out_col] = encoder.transform(df[cat_col], df[target_col])
    return out
