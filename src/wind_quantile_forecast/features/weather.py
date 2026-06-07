"""Operational NWP weather driver features for day-ahead wind forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd
from energy_features.wind import air_density_kg_m3

from wind_quantile_forecast.config import NWP_STANDARD_PRESSURE_PA

WIND_SPEED_HUB_COL = "nwp_wind_speed_hub_mps"
WIND_DIRECTION_COL = "nwp_wind_direction_10m"
WIND_DIRECTION_SIN_COL = "nwp_wind_direction_sin"
WIND_DIRECTION_COS_COL = "nwp_wind_direction_cos"
AIR_DENSITY_COL = "nwp_air_density_kg_m3"
TEMP_COL = "nwp_t2m_k"
PRESSURE_COL = "nwp_surface_pressure_pa"

WEATHER_DRIVER_COLS: tuple[str, ...] = (
    WIND_SPEED_HUB_COL,
    WIND_DIRECTION_SIN_COL,
    WIND_DIRECTION_COS_COL,
    AIR_DENSITY_COL,
)


def add_weather_driver_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive hub wind, wind-direction sin/cos, and air density from NWP covariates.

    Uses ``nwp_surface_pressure_pa`` when present; otherwise falls back to
    ``NWP_STANDARD_PRESSURE_PA`` (GFS merge schema has no surface pressure).
    """
    out = df.copy()

    if WIND_DIRECTION_COL in out.columns:
        radians = np.deg2rad(out[WIND_DIRECTION_COL].astype(float))
        out[WIND_DIRECTION_SIN_COL] = np.sin(radians)
        out[WIND_DIRECTION_COS_COL] = np.cos(radians)
    elif {"nwp_wind_u_10m", "nwp_wind_v_10m"}.issubset(out.columns):
        u = out["nwp_wind_u_10m"].astype(float)
        v = out["nwp_wind_v_10m"].astype(float)
        speed = np.hypot(u, v).clip(lower=1e-6)
        out[WIND_DIRECTION_SIN_COL] = v / speed
        out[WIND_DIRECTION_COS_COL] = u / speed

    if AIR_DENSITY_COL not in out.columns and TEMP_COL in out.columns:
        if PRESSURE_COL in out.columns:
            pressure = out[PRESSURE_COL].astype(float)
        else:
            pressure = pd.Series(
                float(NWP_STANDARD_PRESSURE_PA),
                index=out.index,
                dtype=float,
            )
        out[AIR_DENSITY_COL] = air_density_kg_m3(pressure, out[TEMP_COL].astype(float))

    return out


def weather_driver_columns(df: pd.DataFrame) -> list[str]:
    """Return key weather driver columns that are present in *df*."""
    return [c for c in WEATHER_DRIVER_COLS if c in df.columns]
