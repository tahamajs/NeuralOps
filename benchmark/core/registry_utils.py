"""Registry interaction helpers for querying and inspecting registered attention mechanisms."""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Callable, Optional, List, Any

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
    candidate = ATTENTION_REGISTRY.get(f"{model_name}_inject")
    if candidate is not None and callable(candidate) and not isinstance(candidate, type):
        return True
    candidate = ATTENTION_REGISTRY.get(model_name)
    if candidate is not None and callable(candidate) and not isinstance(candidate, type):
        return True
    return False


def get_inject_function(model_name: str) -> Optional[Callable[..., Any]]:
    """
    Return the registered injection function for the given model_name if available, else None.
    """
    candidate = ATTENTION_REGISTRY.get(f"{model_name}_inject")
    if candidate is not None and callable(candidate) and not isinstance(candidate, type):
        return candidate
    candidate = ATTENTION_REGISTRY.get(model_name)
    if candidate is not None and callable(candidate) and not isinstance(candidate, type):
        return candidate
    return None
