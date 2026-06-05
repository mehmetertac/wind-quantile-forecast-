"""Pre-commit hook: fail if any tracked source file exceeds 1,000 lines."""

import sys
from pathlib import Path

MAX_LINES = 1000
SCAN_DIRS = ["src", "tests", "scripts"]
SCAN_EXTENSIONS = {".py", ".md", ".yaml", ".yml", ".toml"}


def check_file_sizes(root: Path) -> list[str]:
    violations: list[str] = []
    for scan_dir in SCAN_DIRS:
        dir_path = root / scan_dir
        if not dir_path.exists():
            continue
        for file_path in dir_path.rglob("*"):
            if file_path.suffix not in SCAN_EXTENSIONS:
                continue
            line_count = sum(1 for _ in file_path.open(encoding="utf-8", errors="replace"))
            if line_count > MAX_LINES:
                rel = file_path.relative_to(root)
                violations.append(f"{rel}: {line_count} lines (max {MAX_LINES})")
    return violations


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    violations = check_file_sizes(root)
    if violations:
        print("Files exceeding line limit:")
        for v in violations:
            print(f"  {v}")
        print(f"\nRefactor files to stay under {MAX_LINES} lines. See AGENTS.md.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
