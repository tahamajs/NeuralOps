"""Registry interaction helpers for querying and inspecting registered attention mechanisms."""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Callable, Optional, List, Any

# Ensure Codes directory is on sys.path to access framework modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CODES_DIR = PROJECT_ROOT / "Codes"
if str(CODES_DIR) not in sys.path:
    sys.path.insert(0, str(CODES_DIR))

try:
    from framework.core.registry import ATTENTION_REGISTRY
    import framework.attentions  # Trigger auto-discovery and decorator registration
except ImportError:
    ATTENTION_REGISTRY = {}


def get_all_registered_models() -> List[str]:
    """
    Return a sorted list of unique canonical attention model names available in ATTENTION_REGISTRY.
    Filters out internal `_inject` naming suffixes to provide clean model keys.
    """
    keys = set()
    for name in ATTENTION_REGISTRY.keys():
        if name.endswith("_inject"):
            base = name[:-7]
            keys.add(base)
        else:
            keys.add(name)
    keys.add("full")
    return sorted(list(keys))


def has_inject_function(model_name: str) -> bool:
    """
    Check if ATTENTION_REGISTRY contains an injection function for the given model_name.
    """
    if f"{model_name}_inject" in ATTENTION_REGISTRY:
        candidate = ATTENTION_REGISTRY[f"{model_name}_inject"]
        return callable(candidate) and not isinstance(candidate, type)
    if model_name in ATTENTION_REGISTRY:
        candidate = ATTENTION_REGISTRY[model_name]
        return callable(candidate) and not isinstance(candidate, type)
    return False


def get_inject_function(model_name: str) -> Optional[Callable[..., Any]]:
    """
    Return the registered injection function for the given model_name if available, else None.
    """
    if f"{model_name}_inject" in ATTENTION_REGISTRY:
        candidate = ATTENTION_REGISTRY[f"{model_name}_inject"]
        if callable(candidate) and not isinstance(candidate, type):
            return candidate
    if model_name in ATTENTION_REGISTRY:
        candidate = ATTENTION_REGISTRY[model_name]
        if callable(candidate) and not isinstance(candidate, type):
            return candidate
    return None
