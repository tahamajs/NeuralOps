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
        return self._dict.keys()


MODEL_REGISTRY = Registry("models")
ATTENTION_REGISTRY = Registry("attentions")
VISUALIZER_REGISTRY = Registry("visualizers")