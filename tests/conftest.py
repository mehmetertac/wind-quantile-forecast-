"""Shared pytest fixtures."""

import numpy as np
import pytest


@pytest.fixture
def sample_forecast_data() -> tuple[np.ndarray, np.ndarray]:
    """Small synthetic y_true / y_pred arrays for testing."""
    y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    y_pred = np.array([12.0, 18.0, 32.0, 38.0, 52.0])
    return y_true, y_pred
