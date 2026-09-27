#!/usr/bin/env python3
"""Find all occurrences of kwargs.get("rank", ...) that might cause
the 'multiple values for keyword argument' error when rank is also
passed explicitly to the constructor."""
from __future__ import annotations

import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent / "Codes" / "framework"/"compression_engine"
if not BASE.exists():
    # Fallback if run from elsewhere
    for candidate in [
        Path("Codes/framework"),
        Path("D:/SVD_Project/Codes/framework"),
        Path.home() / "Documents/uni/Project/Codes/framework",
    ]:
        if candidate.exists():
            BASE = candidate
            break

PAT = re.compile(r"kwargs\.get\s*\(\s*(['\"])rank\1\s*,")

def main(fix: bool = False) -> int:
    if not BASE.exists():
        print(f"ERROR: {BASE} not found. Run from project root.")
        return 2

    files = list(BASE.glob("**/*.py"))
    hits = []

    for path in files:
        try:
            content = path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"  [skip] {path}: {e}")
            continue

        for m in PAT.finditer(content):
            line_no = content[: m.start()].count("\n") + 1
            hits.append((path, line_no, m.group(0), content))

    if not hits:
        print("NO occurrences of kwargs.get(\"rank\", ...) found.")
        print("All modules use kwargs.pop(), which is safe.")
        return 0

    print(f"FOUND {len(hits)} occurrence(s) of kwargs.get(\"rank\", ...):")
    print()
    for path, line_no, snippet, _ in hits:
        print(f"  {path}:{line_no}")
        print(f"      {snippet}")
    print()

    if not fix:
        print("Run with --fix to automatically replace with kwargs.pop(...).")
        return 1

    # Group hits by file
    by_file: dict[Path, list[int]] = {}
    for path, line_no, _, _ in hits:
        by_file.setdefault(path, []).append(line_no)

    print(f"Patching {len(by_file)} file(s)...")
    for path, _ in by_file.items():
        try:
            content = path.read_text(encoding="utf-8")
            new_content = PAT.sub(
                lambda m: f"kwargs.pop({m.group(1)}rank{m.group(1)},",
                content,
            )
            if new_content != content:
                path.write_text(new_content, encoding="utf-8")
                print(f"  patched: {path}")
        except Exception as e:
            print(f"  FAILED: {path}: {e}")

    print("Done. Re-run without --fix to verify.")
    return 0


if __name__ == "__main__":
    fix = "--fix" in sys.argv
    sys.exit(main(fix=fix))
