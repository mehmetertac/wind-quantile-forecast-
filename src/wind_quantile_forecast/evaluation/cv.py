"""Rolling-origin cross-validation for temporal wind forecasting."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator, TimeSeriesSplit

from wind_quantile_forecast.config import QUANTILES, TARGET_COL

VALID_TIME_COL = "valid_time"
SplitMode = Literal["expanding", "rolling"]


@dataclass(frozen=True)
class TimeFold:
    """One rolling-origin fold defined by train and test timestamp sets."""

    train_times: pd.DatetimeIndex
    test_times: pd.DatetimeIndex
    fold: int

    @property
    def n_train(self) -> int:
        return len(self.train_times)

    @property
    def n_test(self) -> int:
        return len(self.test_times)


@dataclass
class RollingOriginCVResult:
    """Aggregated rolling-origin CV outputs."""

    fold_metrics: pd.DataFrame
    oof_predictions: pd.DataFrame | None = None


def unique_sorted_times(times: pd.DatetimeIndex | pd.Series) -> pd.DatetimeIndex:
    """Return unique, monotonically increasing timestamps."""
    return pd.DatetimeIndex(pd.Series(times).drop_duplicates().sort_values())


def rolling_origin_time_folds(
    times: pd.DatetimeIndex | pd.Series,
    n_splits: int = 5,
    *,
    mode: SplitMode = "expanding",
    max_train_size: int | None = None,
    test_size: int | None = None,
    gap: int = 0,
) -> list[TimeFold]:
    """Build expanding or rolling-origin folds on unique sorted timestamps.

    Each fold trains on timestamps strictly before the test window (minus ``gap``)
    and tests on the next chronological segment. This mirrors operational
    day-ahead evaluation: models never train on future ``valid_time`` rows.

    Args:
        times: Observation timestamps (e.g. ``valid_time``).
        n_splits: Number of test windows (folds).
        mode: ``"expanding"`` grows the train window; ``"rolling"`` caps it at
            ``max_train_size``.
        max_train_size: Maximum train periods for rolling mode (required when
            ``mode="rolling"``).
        test_size: Fixed test-window length in periods; auto-sized when ``None``.
        gap: Periods between the last train timestamp and first test timestamp.

    Returns:
        List of :class:`TimeFold` objects.
    """
    if n_splits < 2:
        msg = f"n_splits must be >= 2, got {n_splits}"
        raise ValueError(msg)
    if gap < 0:
        msg = f"gap must be >= 0, got {gap}"
        raise ValueError(msg)
    if mode == "rolling" and max_train_size is None:
        msg = "max_train_size is required when mode='rolling'"
        raise ValueError(msg)
    if max_train_size is not None and max_train_size < 1:
        msg = f"max_train_size must be >= 1, got {max_train_size}"
        raise ValueError(msg)

    unique = unique_sorted_times(times)
    n = len(unique)
    if n < n_splits + 1:
        msg = f"need at least {n_splits + 1} unique timestamps for {n_splits} folds, got {n}"
        raise ValueError(msg)

    min_train = max(1, n // (n_splits + 1))
    span = test_size if test_size is not None else max(1, (n - min_train) // n_splits)

    folds: list[TimeFold] = []
    for k in range(n_splits):
        test_start = min_train + k * span
        test_end = test_start + span if k < n_splits - 1 else n
        if test_start >= n or test_start >= test_end:
            break

        train_end = test_start - gap
        if train_end < 1:
            continue

        if mode == "expanding":
            train = unique[:train_end]
        else:
            train_start = max(0, train_end - int(max_train_size))
            train = unique[train_start:train_end]

        test = unique[test_start:test_end]
        if len(train) == 0 or len(test) == 0:
            continue
        folds.append(TimeFold(train_times=train, test_times=test, fold=k + 1))

    if not folds:
        msg = "could not construct any rolling-origin folds"
        raise ValueError(msg)
    return folds


class RollingOriginSplit(BaseCrossValidator):
    """Sklearn-compatible rolling-origin splitter for temporal forecasting.

    Supports two backends:

    - ``backend="custom"`` (default): timestamp folds via
      :func:`rolling_origin_time_folds` on a ``time_col`` or ``groups`` array.
    - ``backend="sklearn"``: wraps :class:`sklearn.model_selection.TimeSeriesSplit`
      on row positions (requires one row per sorted timestamp).

    Example::

        splitter = RollingOriginSplit(n_splits=5, mode="expanding")
        for train_idx, test_idx in splitter.split(df, groups=df["valid_time"]):
            ...
    """

    def __init__(
        self,
        n_splits: int = 5,
        *,
        mode: SplitMode = "expanding",
        max_train_size: int | None = None,
        test_size: int | None = None,
        gap: int = 0,
        time_col: str = VALID_TIME_COL,
        backend: Literal["custom", "sklearn"] = "custom",
    ) -> None:
        self.n_splits = n_splits
        self.mode = mode
        self.max_train_size = max_train_size
        self.test_size = test_size
        self.gap = gap
        self.time_col = time_col
        self.backend = backend

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        if X is None and groups is None:
            return self.n_splits
        folds = self._time_folds(self._resolve_times(X, groups))
        return len(folds)

    def split(
        self,
        X,
        y=None,
        groups=None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(train_idx, test_idx)`` row index arrays for each fold."""
        n = len(X)
        if self.backend == "sklearn":
            yield from self._sklearn_splits(n)
            return

        time_values = self._row_time_values(X, groups, n)
        folds = self._time_folds(unique_sorted_times(time_values))

        for fold in folds:
            train_mask = time_values.isin(fold.train_times).to_numpy()
            test_mask = time_values.isin(fold.test_times).to_numpy()
            train_idx = np.flatnonzero(train_mask)
            test_idx = np.flatnonzero(test_mask)
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            yield train_idx, test_idx

    def split_times(
        self,
        times: pd.DatetimeIndex | pd.Series,
    ) -> Iterator[TimeFold]:
        """Yield :class:`TimeFold` objects directly on unique timestamps."""
        yield from self._time_folds(unique_sorted_times(times))

    def split_frame(
        self,
        df: pd.DataFrame,
        *,
        time_col: str | None = None,
        sort: bool = True,
    ) -> Iterator[tuple[pd.DataFrame, pd.DataFrame, TimeFold]]:
        """Yield ``(train_df, test_df, fold)`` triples from a modeling table.

        Args:
            df: Input frame (e.g. day-ahead feature table).
            time_col: Timestamp column; defaults to ``self.time_col``.
            sort: Sort ``df`` by ``time_col`` before splitting.
        """
        col = time_col or self.time_col
        if col not in df.columns:
            msg = f"time column {col!r} not in dataframe"
            raise KeyError(msg)

        work = df.sort_values(col) if sort else df
        for fold in self.split_times(work[col]):
            train = work[work[col].isin(fold.train_times)]
            test = work[work[col].isin(fold.test_times)]
            if train.empty or test.empty:
                continue
            yield train, test, fold

    def _time_folds(self, times: pd.DatetimeIndex) -> list[TimeFold]:
        return rolling_origin_time_folds(
            times,
            self.n_splits,
            mode=self.mode,
            max_train_size=self.max_train_size,
            test_size=self.test_size,
            gap=self.gap,
        )

    def _row_time_values(self, X, groups, n: int) -> pd.Series:
        if groups is not None:
            return pd.Series(groups, index=np.arange(n))
        if isinstance(X, pd.DataFrame) and self.time_col in X.columns:
            return pd.Series(X[self.time_col].to_numpy(), index=np.arange(n))
        msg = f"groups or DataFrame with {self.time_col!r} required for custom backend"
        raise ValueError(msg)

    def _resolve_times(self, X, groups) -> pd.DatetimeIndex:
        n = len(X) if X is not None else 0
        return unique_sorted_times(self._row_time_values(X, groups, n))

    def _sklearn_splits(self, n_samples: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        max_train = self.max_train_size if self.mode == "rolling" else None
        tscv = TimeSeriesSplit(
            n_splits=self.n_splits,
            max_train_size=max_train,
            test_size=self.test_size,
            gap=self.gap,
        )
        dummy = np.zeros((n_samples, 1))
        yield from tscv.split(dummy)


FoldEvaluator = Callable[[pd.DataFrame, pd.DataFrame, TimeFold], dict[str, float]]


def run_rolling_origin_cv(
    df: pd.DataFrame,
    evaluate_fold: FoldEvaluator,
    *,
    time_col: str = VALID_TIME_COL,
    splitter: RollingOriginSplit | None = None,
    n_splits: int = 5,
    mode: SplitMode = "expanding",
    max_train_size: int | None = None,
    test_size: int | None = None,
    gap: int = 0,
    collect_oof: Callable[[pd.DataFrame, pd.DataFrame, TimeFold], pd.DataFrame] | None = None,
) -> RollingOriginCVResult:
    """Run rolling-origin CV with a user-supplied per-fold evaluator.

    This is the evaluation backbone: each fold receives causal train/test frames
    keyed by ``time_col`` (default ``valid_time``). The ``evaluate_fold`` callback
    fits on ``train_df`` and returns a metrics dict for ``test_df``.

    Args:
        df: Modeling table with at least ``time_col``.
        evaluate_fold: ``(train_df, test_df, fold) -> metrics dict``.
        time_col: Timestamp column for origin rolling.
        splitter: Optional pre-built splitter; built from other kwargs when ``None``.
        n_splits: Number of folds when building a default splitter.
        mode: ``"expanding"`` or ``"rolling"`` train window.
        max_train_size: Cap on train periods for rolling mode.
        test_size: Optional fixed test-window length.
        gap: Periods between train and test windows.
        collect_oof: Optional callback returning out-of-fold prediction rows.

    Returns:
        :class:`RollingOriginCVResult` with per-fold metrics (and optional OOF preds).
    """
    cv = splitter or RollingOriginSplit(
        n_splits=n_splits,
        mode=mode,
        max_train_size=max_train_size,
        test_size=test_size,
        gap=gap,
        time_col=time_col,
    )

    fold_rows: list[dict[str, float | int]] = []
    oof_parts: list[pd.DataFrame] = []

    for train_df, test_df, fold in cv.split_frame(df, time_col=time_col):
        metrics = evaluate_fold(train_df, test_df, fold)
        row: dict[str, float | int] = {
            "fold": fold.fold,
            "n_train": fold.n_train,
            "n_test": fold.n_test,
        }
        row.update({k: float(v) for k, v in metrics.items()})
        fold_rows.append(row)
        if collect_oof is not None:
            oof_parts.append(collect_oof(train_df, test_df, fold))

    if not fold_rows:
        msg = "no fold metrics produced — check timestamp coverage and n_splits"
        raise ValueError(msg)

    oof = pd.concat(oof_parts, ignore_index=True) if oof_parts else None
    return RollingOriginCVResult(fold_metrics=pd.DataFrame(fold_rows), oof_predictions=oof)


def default_quantile_fold_metrics(
    y_true: np.ndarray | pd.Series,
    predictions: dict[float, np.ndarray | pd.Series],
    *,
    quantiles: Sequence[float] = QUANTILES,
) -> dict[str, float]:
    """Pinball loss and prediction-interval coverage for one test fold."""
    from wind_quantile_forecast.models.pinball import pinball_loss

    yt = np.asarray(y_true, dtype=float)
    q_lo, q_med, q_hi = quantiles
    y_med = np.asarray(predictions[q_med], dtype=float)
    metrics: dict[str, float] = {
        "mae": float(np.mean(np.abs(yt - y_med))),
        "n_test": float(len(yt)),
    }
    for q in quantiles:
        yq = np.asarray(predictions[q], dtype=float)
        metrics[f"pinball_q{int(q * 100):02d}"] = pinball_loss(yt, yq, q)
    if q_lo in predictions and q_hi in predictions:
        lo = np.asarray(predictions[q_lo], dtype=float)
        hi = np.asarray(predictions[q_hi], dtype=float)
        metrics["pi80_coverage"] = float(np.mean((yt >= lo) & (yt <= hi)))
    return metrics


def evaluate_quantile_origin_cv(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    predict_fold: Callable[[pd.DataFrame, pd.DataFrame], dict[float, np.ndarray]],
    *,
    target_col: str = TARGET_COL,
    time_col: str = VALID_TIME_COL,
    quantiles: Sequence[float] = QUANTILES,
    splitter: RollingOriginSplit | None = None,
    n_splits: int = 5,
) -> RollingOriginCVResult:
    """Rolling-origin CV for quantile models using a ``predict_fold`` callback.

    ``predict_fold(train_df, test_df)`` must fit on train and return a dict of
    quantile -> predictions aligned with ``test_df``.
    """
    cols = list(feature_cols)

    def evaluate_fold(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        fold: TimeFold,
    ) -> dict[str, float]:
        tr = train_df.dropna(subset=[*cols, target_col])
        te = test_df.dropna(subset=[*cols, target_col])
        if tr.empty or te.empty:
            return {"n_test": 0.0}
        preds = predict_fold(tr, te)
        return default_quantile_fold_metrics(te[target_col], preds, quantiles=quantiles)

    def collect_oof(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        fold: TimeFold,
    ) -> pd.DataFrame:
        tr = train_df.dropna(subset=[*cols, target_col])
        te = test_df.dropna(subset=[*cols, target_col])
        if tr.empty or te.empty:
            return pd.DataFrame()
        preds = predict_fold(tr, te)
        out = te[[time_col, target_col]].copy()
        for q, arr in preds.items():
            out[f"pred_q{int(q * 100):02d}"] = np.asarray(arr, dtype=float)
        out["fold"] = fold.fold
        return out

    return run_rolling_origin_cv(
        df,
        evaluate_fold,
        time_col=time_col,
        splitter=splitter,
        n_splits=n_splits,
        collect_oof=collect_oof,
    )
