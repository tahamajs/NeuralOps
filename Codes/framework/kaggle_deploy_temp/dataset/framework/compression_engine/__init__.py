"""
Auto-import all compressor modules, silently skipping those with missing dependencies.
"""

import logging
import os

logger = logging.getLogger(__name__)

__all__ = []

for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith(".py") and file != "__init__.py":
        module_name = file[:-3]
        try:
            __import__(f"framework.compression_engine.{module_name}")
            __all__.append(module_name)
        except ImportError as e:
            logger.warning(f"Skipping compressor module '{module_name}' due to: {e}")
