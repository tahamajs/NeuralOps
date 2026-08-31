"""
Auto-import all attention modules and export their registered classes.
"""

import logging
import os
import sys
import importlib
from framework.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)

__all__ = ["ATTENTION_REGISTRY"]

_curr_dir = os.path.dirname(__file__)
for file in os.listdir(_curr_dir):
    if file.endswith(".py") and not file.startswith(".") and not file.startswith("_"):
        module_name = file[:-3]
        try:
            mod = __import__(f"framework.attentions.{module_name}", fromlist=[module_name])
            __all__.append(module_name)
            for attr in dir(mod):
                if not attr.startswith("_"):
                    val = getattr(mod, attr)
                    if isinstance(val, type):
                        globals()[attr] = val
                        __all__.append(attr)
        except Exception as e:
            logger.warning(f"Skipping attention module '{module_name}' due to: {e}")

# Compatibility for legacy callers that used a capitalized module path on
# case-sensitive systems.  The implementation lives in ``mamba.py``.
try:
    _mamba_mod = importlib.import_module(".mamba", __name__)
    sys.modules.setdefault(f"{__name__}.Mamba", _mamba_mod)
except Exception as e:
    logger.warning("Could not install Mamba compatibility alias: %s", e)
