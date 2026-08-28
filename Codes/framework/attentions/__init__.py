"""
Auto-import all attention modules and export their registered classes.
"""

import logging
import os
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
