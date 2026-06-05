"""Command-line entry point for the wind quantile forecast pipeline."""

import argparse


def main() -> None:
    """Run the full forecasting pipeline: download -> train -> evaluate -> explain."""
    parser = argparse.ArgumentParser(
        description="Day-ahead probabilistic wind power forecast (P10/P50/P90)"
    )
    parser.add_argument(
        "--country",
        default="DE",
        help="ISO country code for OPSD wind data (default: DE)",
    )
    parser.add_argument(
        "--backend",
        choices=["lightgbm", "xgboost", "catboost"],
        default="lightgbm",
        help="GBM backend for quantile regression (default: lightgbm)",
    )
    args = parser.parse_args()
    raise NotImplementedError(
        f"Pipeline not yet implemented. country={args.country}, backend={args.backend}"
    )


if __name__ == "__main__":
    main()
