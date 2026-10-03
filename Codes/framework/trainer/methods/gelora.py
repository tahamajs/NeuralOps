"""
GeLoRA: Intrinsic dimension estimation for adaptive rank selection.
"""
from typing import Any

import logging

import torch
import torch.nn as nn

from .registry import register_method

logger = logging.getLogger(__name__)


def estimate_intrinsic_dimension_pytorch(hidden_states, k=5) -> Any:
    """Estimate intrinsic dimension pytorch."""
    if hidden_states.dim() == 3:
        hidden_states = hidden_states.mean(dim=1)
    elif hidden_states.dim() != 2:
        raise ValueError(f"Expected 2D or 3D tensor, got {hidden_states.dim()}D")
    N = hidden_states.shape[0]
    if k + 1 > N:
        return 1.0
    if N > 1000:
        idx = torch.randperm(N)[:1000]
        hidden_states = hidden_states[idx]
    dist = torch.cdist(hidden_states, hidden_states)
    dist.fill_diagonal_(float("inf"))
    topk_dist, _ = torch.topk(dist, k, dim=1, largest=False)
    d_k = topk_dist[:, -1:]
    log_ratios = torch.log(topk_dist / (d_k + 1e-8))
    id_per_point = -1.0 / torch.mean(log_ratios, dim=1)
    id_global = torch.mean(id_per_point).item()
    return max(1.0, id_global)


def get_encoder_layers(model) -> Any:
    """Return the encoder layers."""
    if hasattr(model, "distilbert") and hasattr(model.distilbert, "transformer"):
        return model.distilbert.transformer.layer
    if hasattr(model, "bert") and hasattr(model.bert, "encoder"):
        return model.bert.encoder.layer
    if hasattr(model, "roberta") and hasattr(model.roberta, "encoder"):
        return model.roberta.encoder.layer
    if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        return model.encoder.layer
    layers = []
    for name, module in model.named_modules():
        if "layer" in name and isinstance(module, (nn.ModuleList, nn.Sequential)):
            for submod in module:
                layers.append(submod)
        elif "layer" in name:
            layers.append(module)
    if layers:
        return layers
    raise ValueError("Cannot find encoder layers")


def collect_hidden_states(model, dataloader, num_batches=10, max_samples=800) -> Any:
    """Collect hidden states."""
    layers = get_encoder_layers(model)
    num_layers = len(layers)
    hidden_states_per_layer = {i: [] for i in range(num_layers)}
    hooks = []

    def hook_fn(layer_idx) -> Any:
        """Hook fn."""
        def hook(module, input, output) -> None:
            """Hook."""
            hs = output[0].detach() if isinstance(output, tuple) else output.detach()
            hidden_states_per_layer[layer_idx].append(hs)

        return hook

    for i, layer in enumerate(layers):
        hook = layer.register_forward_hook(hook_fn(i))
        hooks.append(hook)

    model.eval()
    total = 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches or total >= max_samples:
                break
            batch = {
                k: v.to(model.device)
                for k, v in batch.items()
                if isinstance(v, torch.Tensor)
            }
            model(**batch)
            total += batch["input_ids"].size(0)

    for hook in hooks:
        hook.remove()
    for i in range(num_layers):
        if hidden_states_per_layer[i]:
            hidden_states_per_layer[i] = torch.cat(hidden_states_per_layer[i], dim=0)
        else:
            hidden_states_per_layer[i] = torch.empty(0)
    return hidden_states_per_layer


def compute_gelora_base_ranks(model, dataloader, config) -> Any:
    """Compute and return gelora base ranks."""
    layer_hidden = collect_hidden_states(
        model, dataloader, num_batches=8, max_samples=800
    )
    layer_id = {}
    for idx, hs in layer_hidden.items():
        if hs.shape[0] < 10:
            id_val = 1.0
        else:
            id_val = estimate_intrinsic_dimension_pytorch(hs, k=min(5, hs.shape[0] - 1))
        base_rank = max(
            config.gelora_min_rank,
            min(config.gelora_max_rank, int(id_val + config.gelora_margin)),
        )
        layer_id[idx] = base_rank
        logger.info(f"Layer {idx} intrinsic dim ≈ {id_val:.2f} → base rank {base_rank}")

    target_substrings = ["query", "key", "value", "dense"]
    rank_dict = {}
    for name, _ in model.named_modules():
        if (
            any(ts in name for ts in target_substrings)
            and "classifier" not in name
            and "pooler" not in name
        ):
            parts = name.split(".")
            layer_idx = None
            for i, part in enumerate(parts):
                if part == "layer" and i + 1 < len(parts):
                    try:
                        layer_idx = int(parts[i + 1])
                        break
                    except (ValueError, TypeError):
                        pass
            rank_dict[name] = layer_id.get(layer_idx, config.lora_r)
    return rank_dict


@register_method("fisher_gelora")
def apply_fisher_gelora(model, config) -> None:
    """Apply fisher gelora."""
    logger.info(
        "Fisher‑GeLoRA registered. Use compute_gelora_base_ranks in training script."
    )
