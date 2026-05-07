import importlib
import warnings
warnings.filterwarnings("ignore")

def _safe_import_modules():
    # List the modules that are OK to fail
    modules = [
        "framework.trainer.registry",
        "framework.core.registry",
        "framework.attentions.core.registry",
        "framework.compression_engine.core.registry",
        # ... add any others you want to auto-import,
        # but it's fine to leave this list empty – the lazy imports in main.py will work
    ]
    for m in modules:
        try:
            importlib.import_module(m)
        except Exception:
            pass

_safe_import_modules()
