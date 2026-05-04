"""Efficient Attention Framework package.

Subpackages are imported explicitly by callers to avoid import-time dependency
and registration side effects.
"""

# framework/attentions/__init__.py
import importlib
import pkgutil
import sys
from pathlib import Path

# Directories to skip (non‑attention code or broken dependencies)
# SKIP_DIRS = {'compression', 'Other_Attentions', 'utils', 'core', '__pycache__'}

def _safe_import_all():
    base_pkg = "framework"
    base_path = Path(__file__).parent

    for py_file in base_path.rglob("*.py"):
        if py_file.name == "__init__.py":
            continue
        # Skip files in excluded directories
        # if any(skip in py_file.parts for skip in SKIP_DIRS):
        #     continue
        rel = py_file.relative_to(base_path).with_suffix("")
        parts = rel.parts
        mod_name = ".".join(parts)
        full_name = f"{base_pkg}.{mod_name}"
        try:
            importlib.import_module(full_name)
        except Exception as e:
            print(f"⚠️ Could not import {full_name}: {e}", file=sys.stderr)

_safe_import_all()