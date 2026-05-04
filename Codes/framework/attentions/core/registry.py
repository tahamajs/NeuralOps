# framework/attentions/core/registry.py
class Registry:
    def __init__(self, name):
        self.name = name
        self._dict = {}

    def register(self, key=None):
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

    def __contains__(self, key):
        return key in self._dict


ATTENTION_REGISTRY = Registry("attentions")
AUX_REGISTRY = Registry("auxiliary")   # <-- for RoPE, SSM, DeltaNet, etc.