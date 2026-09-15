#!/usr/bin/env python3
"""
Remove the Qwen3.5 / qwen3_5 model block from all run_all_4_*.py runner scripts.
Handles both naming conventions and nested f-string braces (e.g. {MODULE} inside dicts).
"""
import re
import glob
import os

runners_dir = os.path.join(os.path.dirname(__file__), "Codes", "runners")
pattern = os.path.join(runners_dir, "run_all_4_*.py")
files = sorted(glob.glob(pattern))

print(f"Found {len(files)} runner scripts in {runners_dir}")

# Allows ONE level of nested braces (for f-string vars like {MODULE}):
QWEN_BLOCK = re.compile(
    r',?\s*\{(?:[^{}]|\{[^{}]*\})*?(?:qwen3_5|Qwen3\.5|Qwen3_5)(?:[^{}]|\{[^{}]*\})*?\}',
    re.DOTALL
)

modified = 0
skipped = 0
errors = 0

for fpath in files:
    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read()

    if "qwen3_5" not in content and "Qwen3.5" not in content and "Qwen3_5" not in content:
        skipped += 1
        continue

    new_content = QWEN_BLOCK.sub("", content)

    if new_content != content:
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"  PATCHED: {os.path.basename(fpath)}")
        modified += 1
    else:
        print(f"  WARN (no regex match): {os.path.basename(fpath)}")
        errors += 1

print(f"\nDone. Patched: {modified}, Skipped (no qwen): {skipped}, Warnings: {errors}")
