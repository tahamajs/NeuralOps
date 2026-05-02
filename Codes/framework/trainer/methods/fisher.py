import numpy as np, torch, logging
from torch.utils.data import DataLoader
from .registry import register_method

logger = logging.getLogger(__name__)

def compute_fisher_importance(model, dataset, tokenizer, task_keys, config):
    # ... (same as legacy, but uses config fields)
    # Returns dict of normalized fisher scores per module

@register_method("fisher_lora")
def apply_fisher_lora(model, config, dataset, tokenizer, task_keys):
    scores = compute_fisher_importance(model, dataset, tokenizer, task_keys, config)
    if scores:
        # build per-layer ranks based on scores...
        # call _apply_adapters with per_layer_ranks
    else:
        apply_lora(model, config)