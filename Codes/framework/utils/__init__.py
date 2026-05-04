from . import architecture
from . import checkpoint
from . import config
from . import device
from . import helpers
from . import layer_norm_fix
from . import logger
from . import metrics
from . import profiler
from . import seed
from framework.utils.save_results import save_experiment_results


# framework/utils/__init__.py
import torch.nn as nn
from framework.attentions.core.registry import ATTENTION_REGISTRY

def replace_attention(model, attention_key, **kwargs):
    """
    Replace every attention module in a Hugging Face model with a registered block.

    Args:
        model: A Hugging Face transformer model (e.g., AutoModelForSequenceClassification)
        attention_key: Key in ATTENTION_REGISTRY (e.g., "full", "goldfinch_ahn")
        **kwargs: Additional arguments for the attention block constructor (e.g., rank, window_size)

    Returns:
        Model with attention layers replaced.
    """
    attn_class = ATTENTION_REGISTRY.get(attention_key)
    if attn_class is None:
        raise ValueError(f"Unknown attention key: {attention_key}. Available: {ATTENTION_REGISTRY.keys()}")

    config = getattr(model, 'config', None)

    def _replace_module(module):
        for name, child in module.named_children():
            if 'attention' in name.lower() or 'attn' in name.lower():
                try:
                    new_attn = attn_class(config, **kwargs) if config is not None else attn_class(**kwargs)
                    setattr(module, name, new_attn)
                except Exception as e:
                    print(f"Failed to replace {name}: {e}")
            else:
                _replace_module(child)

    _replace_module(model)
    return model