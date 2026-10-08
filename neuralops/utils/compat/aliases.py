"""Projection attribute alias helpers for cross-architecture attention modules."""

from __future__ import annotations

import torch.nn as nn


def install_projection_aliases(module: nn.Module) -> nn.Module:
    """Install bidirectional attribute aliases across canonical projection names.

    Args:
        module: PyTorch module whose child modules may need alias attributes.

    Returns:
        The modified module with aliases attached.
    """
    from neuralops.core.adapters import CANONICAL_PROJECTION_ALIASES

    for child in module.modules():
        if hasattr(child, "self") and hasattr(child.self, "query"):
            for attr_name in ["query", "key", "value"]:
                val = getattr(child.self, attr_name, None)
                if val is not None:
                    try:
                        object.__setattr__(child, attr_name, val)
                    except (AttributeError, TypeError):
                        pass
        if hasattr(child, "output") and hasattr(child.output, "dense"):
            val = getattr(child.output, "dense", None)
            if val is not None:
                try:
                    object.__setattr__(child, "dense", val)
                    object.__setattr__(child, "out_proj", val)
                    object.__setattr__(child, "out_lin", val)
                    object.__setattr__(child, "o_proj", val)
                except (AttributeError, TypeError):
                    pass

        for group in CANONICAL_PROJECTION_ALIASES.values():
            present_attr = next((name for name in group if hasattr(child, name)), None)
            if present_attr is not None:
                mod = getattr(child, present_attr)
                for alias in group:
                    if not hasattr(child, alias):
                        try:
                            object.__setattr__(child, alias, mod)
                        except (AttributeError, TypeError):
                            pass
    return module
