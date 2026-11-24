"""Clean-clone install smoke check: imports runtime deps and core package logic."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Callable

RUNTIME_IMPORTS: tuple[str, ...] = (
    "numpy",
    "pandas",
    "sklearn",
    "lightgbm",
    "xgboost",
    "catboost",
    "shap",
    "matplotlib",
    "category_encoders",
    "optuna",
    "requests",
    "wind_quantile_forecast",
)


def _import_module(name: str) -> None:
    """Import *name* or raise with a clear message."""
    __import__(name)


def _pinball_sanity() -> None:
    """Exercise pinball loss on a trivial perfect prediction."""
    import numpy as np
    from wind_quantile_forecast.models.pinball import pinball_loss

    y = np.array([100.0])
    pred = np.array([100.0])
    loss = pinball_loss(y, pred, quantile=0.5)
    if not np.isfinite(loss) or float(loss) != 0.0:
        msg = f"unexpected pinball loss for perfect prediction: {loss}"
        raise RuntimeError(msg)


def _wind_forecast_cmd() -> list[str]:
    """Resolve the wind-forecast console script next to the active interpreter."""
    script_dir = Path(sys.executable).resolve().parent
    for name in ("wind-forecast.exe", "wind-forecast"):
        candidate = script_dir / name
        if candidate.is_file():
            return [str(candidate)]
    return ["wind-forecast"]


def _cli_help() -> None:
    """Ensure the wind-forecast console script responds to --help."""
    proc = subprocess.run(
        _wind_forecast_cmd() + ["--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        msg = f"wind-forecast --help failed (exit {proc.returncode}): {proc.stderr}"
        raise RuntimeError(msg)


def run_smoke(
    *,
    import_fn: Callable[[str], None] | None = None,
    run_pinball: bool = True,
    run_cli: bool = True,
) -> None:
    """Run smoke checks used after a wheels-only lockfile install.

    Args:
        import_fn: Optional override for module import (for tests).
        run_pinball: When True, run pinball sanity check.
        run_cli: When True, run CLI --help check.
    """
    importer = import_fn or _import_module
    for name in RUNTIME_IMPORTS:
        importer(name)
    if run_pinball:
        _pinball_sanity()
    if run_cli:
        _cli_help()


def main() -> None:
    """Entry point for CI and documented reproducible install."""
    try:
        run_smoke()
    except Exception as exc:
        print(f"smoke_test failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print("smoke_test ok")


if __name__ == "__main__":
    main()
