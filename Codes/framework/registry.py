"""
Central registry for attention injection functions.
All new attention modules register their injection function here.
"""

class Registry:
    """A simple registry mapping strings to callables."""
    def __init__(self, name):
        self.name = name
        self._dict = {}

    def register(self, key=None):
        """Decorator to register a function."""
        def decorator(obj):
            nonlocal key
            if key is None:
                key = obj.__name__
            self._dict[key] = obj
            return obj
        return decorator

    def get(self, key):
        return self._dict.get(key)

    def keys(self):
        return list(self._dict.keys())


# Global registry for attention injectors
ATTENTION_REGISTRY = Registry("attentions")

# -------------------------------------------------------------------
# Register built‑in attention types (done once at import time)
# The injection functions are imported from the attentions/ folder.
# -------------------------------------------------------------------
def _register_builtins():
    # These imports will not cause circular dependencies because
    # they happen after the registry is created.
    from attentions.spectra_v2 import inject_spectra_v2_into_distilbert
    from attentions.leka_v2 import inject_leka_v2_into_distilbert

    ATTENTION_REGISTRY.register("spectra_v2")(inject_spectra_v2_into_distilbert)
    ATTENTION_REGISTRY.register("leka_v2")(inject_leka_v2_into_distilbert)

    # Full attention (no change)
    def inject_full_attention(layer, model_config, comp_config, layer_idx, total_layers, old_attn):
        pass
    ATTENTION_REGISTRY.register("full")(inject_full_attention)

_register_builtins()