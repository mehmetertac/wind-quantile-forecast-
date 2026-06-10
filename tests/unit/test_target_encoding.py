"""Unit tests for leakage-safe target encoding."""

from __future__ import annotations

import pandas as pd
import pytest
from wind_quantile_forecast.config import HOUR_SEASON_COL
from wind_quantile_forecast.features.target_encoding import (
    TargetEncoder,
    add_hour_season_feature,
    encode_fold,
    season_bucket,
    target_encoding_column,
)


def test_season_bucket_meteorological() -> None:
    assert season_bucket(1) == 0
    assert season_bucket(4) == 1
    assert season_bucket(7) == 2
    assert season_bucket(10) == 3


def test_add_hour_season_feature_has_96_levels() -> None:
    idx = pd.date_range("2019-01-01", periods=24 * 30, freq="h", tz="Europe/Berlin")
    df = pd.DataFrame({"valid_time": idx})
    out = add_hour_season_feature(df)
    assert HOUR_SEASON_COL in out.columns
    assert out[HOUR_SEASON_COL].nunique() <= 96
    assert out[HOUR_SEASON_COL].astype(str).str.contains("_").all()


def test_target_encoder_loo_excludes_own_target() -> None:
    cats = pd.Series(["A"] * 5)
    y = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    enc = TargetEncoder(smoothing=1.0).fit(cats, y)
    loo = enc.transform(cats, y)
    for i, yi in enumerate(y):
        others = float(y.drop(index=i).mean())
        expected = (others * 4 + 1.0 * enc.global_mean_) / (4 + 1.0)
        assert loo[i] == pytest.approx(expected)


def test_encode_fold_test_uses_train_statistics_only() -> None:
    train = pd.DataFrame(
        {
            "hour_season": ["0_0", "0_0", "1_1"],
            "wind_mw": [100.0, 200.0, 300.0],
        }
    )
    test = pd.DataFrame({"hour_season": ["99_3"], "wind_mw": [9999.0]})
    tr, te, col = encode_fold(train, test, "hour_season", "wind_mw", smoothing=1.0)
    assert col == target_encoding_column("hour_season")
    enc = TargetEncoder(smoothing=1.0).fit(train["hour_season"], train["wind_mw"])
    assert te[col].iloc[0] == pytest.approx(enc.global_mean_)
    assert tr[col].notna().all()


def test_encode_fold_unseen_category_falls_back_to_global_mean() -> None:
    train = pd.DataFrame({"site": ["A", "B"], "wind_mw": [10.0, 30.0]})
    test = pd.DataFrame({"site": ["Z"], "wind_mw": [1000.0]})
    _, te, col = encode_fold(train, test, "site", "wind_mw")
    assert te[col].iloc[0] == pytest.approx(20.0)


def test_target_encoder_transform_without_target_matches_mapping() -> None:
    cats = pd.Series(["a", "b", "a"])
    y = pd.Series([1.0, 3.0, 5.0])
    enc = TargetEncoder(smoothing=0.0).fit(cats, y)
    mapped = enc.transform(cats)
    assert mapped[0] == pytest.approx(3.0)
    assert mapped[1] == pytest.approx(3.0)
