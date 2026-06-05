"""Unit tests for pinball (quantile) loss."""

import numpy as np
import pytest
from wind_quantile_forecast.models.pinball import pinball_loss


def test_perfect_prediction_zero_loss():
    y = np.array([1.0, 2.0, 3.0])
    loss = pinball_loss(y, y, quantile=0.5)
    assert loss == pytest.approx(0.0)


def test_asymmetric_penalty_overestimate():
    """Overestimating is penalized more at low quantiles."""
    y_true = np.array([10.0])
    y_over = np.array([15.0])  # overestimate by 5
    y_under = np.array([5.0])  # underestimate by 5

    loss_over_q10 = pinball_loss(y_true, y_over, quantile=0.1)
    loss_under_q10 = pinball_loss(y_true, y_under, quantile=0.1)
    assert loss_over_q10 > loss_under_q10


def test_asymmetric_penalty_underestimate():
    """Underestimating is penalized more at high quantiles."""
    y_true = np.array([10.0])
    y_over = np.array([15.0])
    y_under = np.array([5.0])

    loss_over_q90 = pinball_loss(y_true, y_over, quantile=0.9)
    loss_under_q90 = pinball_loss(y_true, y_under, quantile=0.9)
    assert loss_under_q90 > loss_over_q90


def test_median_symmetric_error(sample_forecast_data):
    y_true, y_pred = sample_forecast_data
    loss = pinball_loss(y_true, y_pred, quantile=0.5)
    assert loss > 0.0
    assert isinstance(loss, float)
