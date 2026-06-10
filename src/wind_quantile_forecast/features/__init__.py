"""Feature engineering for wind power forecasting."""

from wind_quantile_forecast.features.calendar import (
    add_calendar_features,
    calendar_feature_columns,
)
from wind_quantile_forecast.features.lags import add_lag_features
from wind_quantile_forecast.features.matrix import (
    assemble_feature_matrix,
    validate_no_target_leakage,
)
from wind_quantile_forecast.features.target_encoding import (
    CatEncoding,
    TargetEncoder,
    add_hour_season_feature,
    add_target_encoding,
    encode_fold,
    target_encoding_column,
)
from wind_quantile_forecast.features.weather import (
    add_weather_driver_features,
    weather_driver_columns,
)

__all__ = [
    "add_calendar_features",
    "add_lag_features",
    "CatEncoding",
    "TargetEncoder",
    "add_hour_season_feature",
    "add_target_encoding",
    "encode_fold",
    "target_encoding_column",
    "add_weather_driver_features",
    "assemble_feature_matrix",
    "calendar_feature_columns",
    "validate_no_target_leakage",
    "weather_driver_columns",
]
