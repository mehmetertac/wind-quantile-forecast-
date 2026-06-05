"""SHAP-based feature importance and explanation plots."""

from pathlib import Path

import pandas as pd


def explain_model(
    model: object,
    X: pd.DataFrame,
    output_dir: Path | None = None,
) -> None:
    """Generate SHAP summary and dependence plots for a trained quantile model.

    Args:
        model: Fitted GBM model (any backend).
        X: Feature matrix used for explanation.
        output_dir: Directory to save SHAP plots. Defaults to reports/figures/.

    Raises:
        NotImplementedError: SHAP explanation not yet implemented.
    """
    raise NotImplementedError("SHAP explanation not yet implemented")
