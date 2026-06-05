"""Quantile regression models."""

from wind_quantile_forecast.models.pinball import pinball_loss
from wind_quantile_forecast.models.quantile_gbm import QuantileGBM

__all__ = ["pinball_loss", "QuantileGBM"]
