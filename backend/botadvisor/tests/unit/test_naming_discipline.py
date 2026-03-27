from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCAN_ROOTS = (REPO_ROOT / "app", REPO_ROOT / "scripts", REPO_ROOT / "tests")
BANNED_EXACT_FILE_STEMS = {"helper", "helpers", "utils", "manager", "processor"}
BANNED_NAME_PARTS = ("helper", "helpers", "utils", "manager", "processor")


def iter_python_files() -> list[Path]:
    files: list[Path] = []
    for root in SCAN_ROOTS:
        files.extend(sorted(root.rglob("*.py")))
    return files


def name_violations(path: Path) -> list[str]:
    violations: list[str] = []
    if path.stem in BANNED_EXACT_FILE_STEMS:
        violations.append(f"file stem '{path.stem}' is too vague")

    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(module):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            lowered = node.name.lower()
            if any(part in lowered for part in BANNED_NAME_PARTS):
                violations.append(f"{type(node).__name__} '{node.name}' uses a banned vague term")
    return violations


def test_naming_discipline_disallows_vague_generic_names():
    violations: list[str] = []
    for path in iter_python_files():
        for violation in name_violations(path):
            violations.append(f"{path.relative_to(REPO_ROOT)}: {violation}")

    assert not violations, "Naming discipline violations found:\n" + "\n".join(violations)
