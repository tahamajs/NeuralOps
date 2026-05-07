import torch
import torch.nn as nn
import numpy as np
from ...attentions.Other_Attentions.compressed_attention import CompressedAttentionLayer
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, METHOD_REGISTRY


class AdaptiveTransformerCompressor:
    def __init__(self, energy_threshold: float = 0.95, min_rank: int = 16):
        self.energy_threshold = energy_threshold
        self.min_rank = min_rank

    def compress_model(self, model: nn.Module):
        print(f"🔧 Starting Compression (Threshold: {self.energy_threshold}, Min Rank: {self.min_rank})")
        compression_stats = []
        for name, module in model.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                try:
                    compressed = CompressedAttentionLayer(module, self.energy_threshold, self.min_rank)
                    # Replace in place
                    parent_name = '.'.join(name.split('.')[:-1])
                    child_name = name.split('.')[-1]
                    parent = model.get_submodule(parent_name) if parent_name else model
                    setattr(parent, child_name, compressed)

                    orig_params = sum(p.numel() for p in module.parameters())
                    new_params = sum(p.numel() for p in compressed.parameters())
                    compression_stats.append({
                        'layer': name,
                        'original_params': orig_params,
                        'compressed_params': new_params,
                        'ratio': new_params / orig_params
                    })
                except Exception as e:
                    print(f"   ❌ Failed to compress {name}: {e}")
        orig_total = sum(p.numel() for p in model.parameters())
        new_total = sum(p.numel() for p in model.parameters())  # after replacement
        total_ratio = new_total / orig_total if orig_total > 0 else 1.0
        print(f"✅ Compression Complete! Total Params: {orig_total} -> {new_total} (Ratio: {total_ratio:.2%})")
        return model, compression_stats


# ----------------------------------------------------------------------
# Entropy‑based compressor registration
# ----------------------------------------------------------------------
def _compute_singular_value_entropy(S: torch.Tensor) -> float:
    """Compute normalized entropy of singular values."""
    p = S / S.sum()
    entropy = -(p * torch.log(p + 1e-8)).sum()
    return entropy.item()


def entropy_budget_compress(model: nn.Module, comp_config) -> nn.Module:
    """
    Apply entropy‑based budget allocation:
    - For each MultiheadAttention layer, compute entropy of its QKV singular values.
    - Layers with higher entropy get higher rank (more capacity),
      layers with lower entropy get lower rank.
    - Uses `comp_config.max_seq_len` as the maximum sequence length for compression.
    """
    total_entropy = 0.0
    layer_entropies = []
    for name, module in model.named_modules():
        if isinstance(module, nn.MultiheadAttention):
            w = module.in_proj_weight.data.cpu().float()
            _, S, _ = torch.linalg.svd(w, full_matrices=False)
            entropy = _compute_singular_value_entropy(S)
            layer_entropies.append((name, module, entropy))
            total_entropy += entropy

    if total_entropy == 0:
        return model

    # Budget parameters from comp_config (can add to CompressionConfig)
    min_rank = getattr(comp_config, 'min_rank', 4)
    max_rank = getattr(comp_config, 'max_rank', 128)
    base_rank = getattr(comp_config, 'base_rank', 32)
    entropy_budget = getattr(comp_config, 'entropy_budget', 1.0)  # default scale

    # Assign rank proportionally to entropy
    for name, module, entropy in layer_entropies:
        normalized_entropy = entropy / total_entropy
        # rank = base_rank * (1 + entropy_budget * (normalized_entropy - 1/len))
        # More entropy → higher rank
        rank = int(base_rank * (1 + entropy_budget * (normalized_entropy - 1.0/len(layer_entropies))))
        rank = max(min_rank, min(rank, max_rank))

        # Compress this attention layer
        compressed = CompressedAttentionLayer(module, rank, rank, method="svd")
        parent_name = '.'.join(name.split('.')[:-1])
        child_name = name.split('.')[-1]
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, compressed)
        print(f"   Compressed {name} with entropy {entropy:.3f} → rank {rank}")

    return model


# Register the compressor
@COMPRESSOR_REGISTRY.register("entropy_budget")
def entropy_budget_compressor_entry(model: nn.Module, comp_config) -> nn.Module:
    """Entry point for the registry."""
    return entropy_budget_compress(model, comp_config)