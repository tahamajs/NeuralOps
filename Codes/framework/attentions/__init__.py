# framework/attentions/__init__.py
import importlib
import pkgutil
import sys
from pathlib import Path

# Directories to skip (they contain non‑attention code, e.g., compression, utils)
# SKIP_DIRS = {'compression', 'Other_Attentions', 'utils', 'core', '__pycache__'}
# SKIP_DIRS = {'hybrid_ssm', 'compression', 'Other_Attentions', 'utils', 'core', '__pycache__'}
def _safe_import_all():
    """Import all Python modules under framework.attentions, skipping known non‑attention dirs and ignoring errors."""
    base_pkg = "framework.attentions"
    base_path = Path(__file__).parent

    for py_file in base_path.rglob("*.py"):
        if py_file.name == "__init__.py":
            continue
        # Skip files inside excluded directories
        # if any(skip in py_file.parts for skip in SKIP_DIRS):
        #     continue
        # Convert path to module name, e.g., "nova.nova"
        rel = py_file.relative_to(base_path).with_suffix("")
        parts = rel.parts
        mod_name = ".".join(parts)
        full_name = f"{base_pkg}.{mod_name}"
        try:
            importlib.import_module(full_name)
        except Exception as e:
            # Print a warning but don't crash the whole process
            print(f"⚠️ Could not import {full_name}: {e}", file=sys.stderr)

_safe_import_all()