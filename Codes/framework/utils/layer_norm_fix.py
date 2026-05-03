"""Fix DistilBERT checkpoint LayerNorm naming (beta/gamma -> bias/weight)."""

import torch.nn as nn


def fix_layernorm_naming(model: nn.Module):
    """
    Rename LayerNorm parameters from 'beta'/'gamma' to 'bias'/'weight'
    for compatibility with HuggingFace checkpoints.

    Args:
        model: PyTorch model to fix (in-place)
    """
    for module in model.modules():
        if isinstance(module, nn.LayerNorm):
            state = dict(module.named_parameters())
            if 'beta' in state and 'bias' not in state:
                bias = module._parameters.pop('beta')
                module.register_parameter('bias', bias)
            if 'gamma' in state and 'weight' not in state:
                weight = module._parameters.pop('gamma')
                module.register_parameter('weight', weight)