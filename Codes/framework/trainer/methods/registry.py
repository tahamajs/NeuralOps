"""
Central registry for training methods (LoRA, Fisher, etc.).
Each entry maps a string name to a callable (model, config, ...).
"""
from typing import Any

METHOD_REGISTRY = {}


def register_method(name: str) -> Any:
    """Decorator to register a method callable."""

    def decorator(fn) -> Any:
        """Decorator."""
        METHOD_REGISTRY[name] = fn
        return fn

    return decorator


def get_method(name: str) -> Any:
    """Return the method function by name, or None if not found."""
    return METHOD_REGISTRY.get(name)


def available_methods() -> Any:
    """Return list of all registered method names."""
    return list(METHOD_REGISTRY.keys())
