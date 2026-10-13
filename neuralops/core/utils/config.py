"""Configuration extraction helpers."""
from __future__ import annotations

from typing import Any

__all__ = ["_cfg_get"]


def _cfg_get(config_or_key: Any, key_or_default: Any = None, default: Any = None) -> Any:
    """Safely extract a configuration attribute.
    
    Supports both signatures:
      1. Standard: _cfg_get(config, key, default=None)
      2. Key-first / single-arg: _cfg_get(key, default=None) -> returns default
    """
    # If first arg is a string and second is not a string, it was called as _cfg_get('key', default)
    if isinstance(config_or_key, str):
        # Called without a config object, return default
        return key_or_default

    config = config_or_key
    key = str(key_or_default) if key_or_default is not None else ""
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)
