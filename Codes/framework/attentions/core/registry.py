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

    def __len__(self):
        return len(self._dict)

    def __repr__(self):
        return f"Registry('{self.name}', entries={list(self._dict.keys())})"


# Core registries for the efficient attention framework
ATTENTION_REGISTRY = Registry("attentions")      # attention modules (Nexus, LEKA, AURORA, etc.)
AUX_REGISTRY = Registry("auxiliary")             # auxiliary modules (RoPE, SSM, DeltaNet, token pruning, etc.)
COMPRESSOR_REGISTRY = Registry("compressors")    # compression strategies (SVD, SLiM, NMF, etc.)
METHOD_REGISTRY = Registry("methods")            # training methods (LoRA, R‑Drop, R3F, Fisher, etc.)

# Optional (if your framework grows)
MODEL_REGISTRY = Registry("models")              # different base models (DistilBERT, BERT, RoBERTa)
DATASET_REGISTRY = Registry("datasets")          # GLUE, SuperGLUE, custom tasks
LOSS_REGISTRY = Registry("losses")               # custom loss functions