"""GeLoRA intrinsic dimension estimation and rank computation."""
import torch, logging, math
from torch.utils.data import DataLoader
from .registry import register_method

logger = logging.getLogger(__name__)

def estimate_intrinsic_dimension_pytorch(hidden_states, k=5):
    # ... (legacy code)

def compute_gelora_base_ranks(model, dataloader, config):
    # ... (legacy code)

@register_method("fisher_gelora")
def apply_fisher_gelora(model, config, dataset, tokenizer, task_keys):
    # For practical purposes, compute Fisher + GeLoRA and set adaptive ranks
    pass