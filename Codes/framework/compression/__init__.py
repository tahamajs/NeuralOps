# framework/compression/__init__.py
import importlib
import pkgutil
from pathlib import Path

# Re‑export the registry and base class
from .core.registry import COMPRESSOR_REGISTRY
from .core.base import BaseCompressor

__all__ = ["COMPRESSOR_REGISTRY", "BaseCompressor"]

# Auto‑import all submodules so that compressors register themselves
def _import_all_submodules(package_name, package_path):
    for info in pkgutil.iter_modules([str(package_path)]):
        module_name = f"{package_name}.{info.name}"
        try:
            importlib.import_module(module_name)
        except Exception:
            pass  # skip failing imports
        if info.ispkg:
            sub_dir = package_path / info.name
            _import_all_submodules(module_name, sub_dir)

_current_path = Path(__file__).parent
_import_all_submodules(__name__, _current_path)