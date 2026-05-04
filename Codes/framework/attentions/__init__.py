# framework/attentions/__init__.py
import importlib
import pkgutil
import sys
from pathlib import Path

# Re‑export the registry and base for convenience
from .core.registry import ATTENTION_REGISTRY
from .core.base import BaseAttention

__all__ = ["ATTENTION_REGISTRY", "BaseAttention"]

def _safe_import_all():
    base_pkg = "framework.attentions"
    base_path = Path(__file__).parent
    for py_file in base_path.rglob("*.py"):
        if py_file.name == "__init__.py":
            continue
        rel = py_file.relative_to(base_path).with_suffix("")
        mod_name = ".".join(rel.parts)
        full_name = f"{base_pkg}.{mod_name}"
        try:
            importlib.import_module(full_name)
        except Exception as e:
            print(f"⚠️ Could not import {full_name}: {e}", file=sys.stderr)

_safe_import_all()