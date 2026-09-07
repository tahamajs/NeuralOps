#!/usr/bin/env python3
"""Verify every Python script referenced in README.md actually exists in the repo.

Extracts ``.py`` filenames from README code fences (bash blocks), resolves each
against the repository, and exits non-zero listing any that are missing. Run in
CI next to the docs build so the README can never drift from reality again:

    .venv/bin/python scripts/check_readme_refs.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
README = PROJECT_ROOT / "README.md"


def referenced_scripts() -> list[str]:
    text = README.read_text(encoding="utf-8")
    fences = re.findall(r"```(?:bash|shell|sh)\n(.*?)```", text, flags=re.DOTALL)
    names: set[str] = set()
    for fence in fences:
        for token in fence.split():
            # match the basename of any *.py mention, however it is pathed
            if token.endswith(".py"):
                names.add(token)
    return sorted(names)


def resolves(token: str) -> bool:
    candidates = [
        PROJECT_ROOT / token,
        PROJECT_ROOT / "Codes" / "framework" / token.removeprefix("Codes/framework/"),
        PROJECT_ROOT / "Codes" / token.removeprefix("Codes/"),
    ]
    if any(c.exists() for c in candidates):
        return True
    # bare filename: search the repo once
    hits = [p for p in PROJECT_ROOT.rglob(Path(token).name) if ".git" not in p.parts and ".venv" not in p.parts]
    return bool(hits)


def main() -> int:
    scripts = referenced_scripts()
    missing = [s for s in scripts if not resolves(s)]
    print(f"README references {len(scripts)} Python script(s); checking existence...")
    for s in scripts:
        status = "OK" if s not in missing else "MISSING"
        print(f"  [{status}] {s}")
    if missing:
        print(f"\nFAIL: {len(missing)} referenced script(s) do not exist in the repo.")
        return 1
    print("\nPASS: all referenced scripts exist.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
