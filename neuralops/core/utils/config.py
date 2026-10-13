"""Configuration extraction helpers."""
from __future__ import annotations

from typing import Any

__all__ = ["_cfg_get"]


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    """Safely extract a configuration attribute from a config object or dict."""
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)
