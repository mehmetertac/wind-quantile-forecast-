"""Data ingestion and preprocessing."""

from wind_quantile_forecast.data.download import download_opsd_wind_data
from wind_quantile_forecast.data.preprocess import preprocess_wind_data

__all__ = ["download_opsd_wind_data", "preprocess_wind_data"]
