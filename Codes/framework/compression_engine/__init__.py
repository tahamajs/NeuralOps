"""
Auto-import all compression_engine modules, silently skipping those with missing dependencies.
"""

import logging
import os
from framework.core.registry import COMPRESSOR_REGISTRY

logger = logging.getLogger(__name__)

__all__ = ["COMPRESSOR_REGISTRY"]

for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith(".py") and not file.startswith(".") and not file.startswith("_"):
        module_name = file[:-3]
        try:
            __import__(f"framework.compression_engine.{module_name}")
            __all__.append(module_name)
        except Exception as e:
            logger.warning(f"Skipping compression_engine module '{module_name}' due to: {e}")
