"""
LLRD: Layer‑wise Learning Rate Decay.
"""

import logging
from typing import Any

from .registry import register_method

logger = logging.getLogger(__name__)


def get_llrd_param_groups(model, base_lr, weight_decay, decay) -> Any:
    """Return the llrd param groups."""
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight", "layernorm.weight"]
    max_layer = -1
    for name, _ in model.named_parameters():
        if "encoder.layer." in name:
            layer = int(name.split("encoder.layer.")[1].split(".")[0])
            max_layer = max(max_layer, layer)
    num_layers = max_layer + 1

    emb_decay, emb_nodecay = [], []
    layer_decay = {i: [] for i in range(num_layers)}
    layer_nodecay = {i: [] for i in range(num_layers)}
    head_decay, head_nodecay = [], []

    for name, param in model.named_parameters():
        if "embeddings" in name:
            if any(nd in name for nd in no_decay):
                emb_nodecay.append(param)
            else:
                emb_decay.append(param)
        elif "encoder.layer." in name:
            layer = int(name.split("encoder.layer.")[1].split(".")[0])
            if any(nd in name for nd in no_decay):
                layer_nodecay[layer].append(param)
            else:
                layer_decay[layer].append(param)
        elif "classifier" in name or "pooler" in name:
            if any(nd in name for nd in no_decay):
                head_nodecay.append(param)
            else:
                head_decay.append(param)

    groups = []
    emb_lr = base_lr * (decay ** (num_layers + 1))
    if emb_decay:
        groups.append({"params": emb_decay, "lr": emb_lr, "weight_decay": weight_decay})
    if emb_nodecay:
        groups.append({"params": emb_nodecay, "lr": emb_lr, "weight_decay": 0.0})

    for layer in range(num_layers):
        lr_layer = base_lr * (decay ** (num_layers - layer))
        if layer_decay[layer]:
            groups.append(
                {
                    "params": layer_decay[layer],
                    "lr": lr_layer,
                    "weight_decay": weight_decay,
                }
            )
        if layer_nodecay[layer]:
            groups.append({"params": layer_nodecay[layer], "lr": lr_layer, "weight_decay": 0.0})

    if head_decay:
        groups.append({"params": head_decay, "lr": base_lr, "weight_decay": weight_decay})
    if head_nodecay:
        groups.append({"params": head_nodecay, "lr": base_lr, "weight_decay": 0.0})
    return groups


@register_method("llrd")
def apply_llrd(model, config) -> None:
    """Apply llrd."""
    logger.info("LLRD registered. Use --method llrd to enable layer‑wise learning rates.")
