"""Tests for scripts/smoke_test.py (no energy_features imports)."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SMOKE_SCRIPT = _REPO_ROOT / "scripts" / "smoke_test.py"

_RUNTIME_SMOKE_MARKERS = ("category_encoders", "lightgbm", "catboost")


def _runtime_smoke_deps_available() -> bool:
    """True when locked runtime packages are importable (same set as smoke_test)."""
    for name in _RUNTIME_SMOKE_MARKERS:
        try:
            __import__(name)
        except ModuleNotFoundError:
            return False
    return True


_runtime_smoke = pytest.mark.skipif(
    not _runtime_smoke_deps_available(),
    reason="install requirements.lock runtime deps to run smoke integration tests",
)


def _load_smoke_module() -> ModuleType:
    """Load smoke_test.py as a module (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("smoke_test", _SMOKE_SCRIPT)
    if spec is None or spec.loader is None:
        msg = f"cannot load {_SMOKE_SCRIPT}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@_runtime_smoke
def test_run_smoke_happy_path() -> None:
    smoke = _load_smoke_module()
    smoke.run_smoke()


def test_run_smoke_missing_module_raises() -> None:
    smoke = _load_smoke_module()

    def failing_import(name: str) -> None:
        if name == "numpy":
            raise ModuleNotFoundError(name)
        __import__(name)

    with pytest.raises(ModuleNotFoundError):
        smoke.run_smoke(import_fn=failing_import)


@_runtime_smoke
def test_smoke_script_main_exit_zero_when_ok() -> None:
    proc = subprocess.run(
        [sys.executable, str(_SMOKE_SCRIPT)],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
