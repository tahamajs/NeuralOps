"""
Auto-import all attention modules, silently skipping those with missing dependencies.
"""

import logging
import os
from framework.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)

__all__ = ["ATTENTION_REGISTRY"]

for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith(".py") and file != "__init__.py":
        module_name = file[:-3]
        try:
            __import__(f"framework.attentions.{module_name}")
            __all__.append(module_name)
        except Exception as e:
            # Module has missing dependencies or syntax issues – skip but warn
            logger.warning(f"Skipping attention module '{module_name}' due to: {e}")
