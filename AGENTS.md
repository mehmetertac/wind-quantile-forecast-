# AGENTS.md — Contributor and Agent Governance

Rules for human contributors and AI agents working in this repository.

## 1. File size limit

No source file should exceed **1,000 lines**. If a file approaches or exceeds this limit:

- Split it into focused modules (e.g. separate `download.py` / `preprocess.py` instead of one `data.py`).
- Extract shared logic into a `utils/` or `common/` submodule.
- A pre-commit hook (`scripts/check_file_size.py`) enforces this automatically.

## 2. Documentation before push

Update documentation **before every push** to the repository:

- **README.md** — keep setup instructions, pipeline description, and deliverables checklist current.
- **Module docstrings** — every public function and class must have a docstring describing purpose, args, and return value.
- **AGENTS.md** — update if governance rules change.
- If you add a CLI flag, config option, or new module, document it in the README or relevant docstring in the same change.

## 3. Tests for every change

Always create at least **minimal unit tests**, even for small changes:

- New function → at least one unit test covering the happy path and one edge case.
- Bug fix → a regression test that would have caught the bug.
- New module → unit tests in `tests/unit/`.
- Pipeline stages → integration tests in `tests/integration/` once the stage is wired end-to-end.

Test naming convention: `test_<what>_<condition>` (e.g. `test_pinball_loss_perfect_prediction`).

## 4. Run tests before commit / push

Tests must pass before code is committed or pushed:

- **Pre-commit hook** runs `pytest -q` and `ruff check` on every commit.
- **File-size hook** runs `scripts/check_file_size.py` on every commit.
- If hooks are not yet installed, run `pre-commit install` after cloning.
- If hooks do not exist yet, create them (see `.pre-commit-config.yaml`).

Manual fallback if hooks are unavailable:

```powershell
pytest -q
ruff check src tests
py scripts/check_file_size.py
```

## Quick reference

| Action | Requirement |
|--------|-------------|
| New file > 1,000 lines | Refactor before committing |
| Any code change | Add/update tests |
| Any code change | Update docs if behavior or API changed |
| Before commit | `pre-commit` hooks pass (pytest + ruff + file-size) |
| Before push | Docs current, all tests green |
