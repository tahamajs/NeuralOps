# framework/models_conf/loaders/__init__.py
import torch.nn as nn
from transformers import AutoModel, AutoConfig

# Registry (can be empty or extended later)
MODEL_LOADER_REGISTRY = {}

def load_model(model_name: str, device: str = "cuda", **kwargs) -> nn.Module:
    """
    Load a HuggingFace model – simple fallback implementation.
    """
    config = AutoConfig.from_pretrained(model_name, **kwargs)
    # Heuristic: if it's a decoder-only model (GPT, LLaMA, etc.)
    if getattr(config, "is_decoder", False):
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    else:
        model = AutoModel.from_pretrained(model_name, **kwargs)
    return model.to(device)

# Optional: register the default loader
MODEL_LOADER_REGISTRY["auto"] = load_model   # or a class, but this is fine