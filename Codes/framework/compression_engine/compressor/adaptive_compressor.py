"""
Adaptive zero‑shot compression for transformer models using SVD.
Automatically determines compression ranks per layer based on singular value energy.
Supports: 'svd_adaptive', 'uniform', 'layer_wise_threshold'
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional


# Attempt to import the compressed attention layer from the framework
try:
    from framework.attentions.compressed_attention import CompressedAttentionLayer
except ImportError:
    # Fallback: define a minimal version if framework not available (for standalone use)
    class CompressedAttentionLayer(nn.Module):
        def __init__(self, original_layer: nn.MultiheadAttention, rank_qkv: int, rank_out: int, method: str = "svd"):
            super().__init__()
            # Minimal placeholder – actual implementation in framework
            self.original_dim = original_layer.embed_dim
            self.num_heads = original_layer.num_heads
            self.head_dim = self.original_dim // self.num_heads
            # ... simplified for fallback
            raise NotImplementedError("Full CompressedAttentionLayer not imported. Please ensure framework is in PYTHONPATH.")


class AdaptiveSVDCompressor:
    """
    Compresses MultiheadAttention layers using SVD with per‑layer rank selection
    based on energy threshold or fixed rank.
    """
    def __init__(self, method: str = "svd_adaptive", energy_threshold: float = 0.95,
                 fixed_rank: Optional[int] = None, min_rank: int = 4, max_rank: int = 128):
        """
        Args:
            method: 'svd_adaptive' (energy threshold), 'uniform' (fixed rank), 'layer_wise_threshold'
            energy_threshold: fraction of singular value energy to retain (0..1)
            fixed_rank: if method='uniform', use this rank for all layers
            min_rank: minimum allowed rank
            max_rank: maximum allowed rank
        """
        self.method = method
        self.energy_threshold = energy_threshold
        self.fixed_rank = fixed_rank
        self.min_rank = min_rank
        self.max_rank = max_rank

    def _rank_from_energy(self, S: np.ndarray) -> int:
        """Compute rank needed to preserve `energy_threshold` of variance."""
        if len(S) == 0:
            return self.min_rank
        total = np.sum(S ** 2)
        cum_energy = np.cumsum(S ** 2) / total
        rank = np.searchsorted(cum_energy, self.energy_threshold) + 1
        return int(np.clip(rank, self.min_rank, self.max_rank))

    def compress_model(self, model: nn.Module) -> Tuple[nn.Module, List[Dict], float]:
        """
        Compress all MultiheadAttention layers in the model.

        Returns:
            model: modified model with replaced attention layers
            stats: list of dicts with compression info per layer
            reduction_percent: total parameter reduction (percent)
        """
        orig_total_params = sum(p.numel() for p in model.parameters())
        stats = []

        # Collect all MultiheadAttention layers
        layers_to_compress = []
        for name, module in model.named_modules():
            if not isinstance(module, nn.MultiheadAttention):
                continue

            # Get SVD of QKV and output weights
            w_qkv = module.in_proj_weight.data.cpu().numpy()
            w_out = module.out_proj.weight.data.cpu().numpy()
            _, S_qkv, _ = np.linalg.svd(w_qkv, full_matrices=False)
            _, S_out, _ = np.linalg.svd(w_out, full_matrices=False)

            layers_to_compress.append({
                'name': name,
                'module': module,
                'S_qkv': S_qkv,
                'S_out': S_out,
            })

        if not layers_to_compress:
            print("No MultiheadAttention layers found.")
            return model, [], 0.0

        # Determine ranks per layer according to method
        for info in layers_to_compress:
            if self.method == 'uniform':
                r_qkv = self.fixed_rank if self.fixed_rank is not None else 32
                r_out = r_qkv
            elif self.method == 'layer_wise_threshold':
                r_qkv = self._rank_from_energy(info['S_qkv'])
                r_out = self._rank_from_energy(info['S_out'])
            else:  # 'svd_adaptive' (default)
                r_qkv = self._rank_from_energy(info['S_qkv'])
                r_out = self._rank_from_energy(info['S_out'])
            info['r_qkv'] = r_qkv
            info['r_out'] = r_out

        # Replace layers
        compressed_count = 0
        for info in layers_to_compress:
            try:
                new_layer = CompressedAttentionLayer(
                    info['module'],
                    info['r_qkv'],
                    info['r_out'],
                    method='svd'   # method used inside CompressedAttentionLayer
                )
                # Replace the module in the model
                parent_path = info['name'].split('.')
                parent = model
                for part in parent_path[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, parent_path[-1], new_layer)

                orig_params = sum(p.numel() for p in info['module'].parameters())
                new_params = sum(p.numel() for p in new_layer.parameters())
                stats.append({
                    'layer': info['name'],
                    'rank_qkv': info['r_qkv'],
                    'rank_out': info['r_out'],
                    'original_params': orig_params,
                    'compressed_params': new_params,
                    'reduction': (1 - new_params / orig_params) * 100
                })
                compressed_count += 1
            except Exception as e:
                print(f"❌ Failed to compress {info['name']}: {e}")

        new_total_params = sum(p.numel() for p in model.parameters())
        reduction_percent = (orig_total_params - new_total_params) / orig_total_params * 100

        print(f"✅ Compressed {compressed_count}/{len(layers_to_compress)} layers")
        print(f"📉 Parameter reduction: {reduction_percent:.2f}%")
        return model, stats, reduction_percent


def get_adaptive_ranks_from_patterns(model: nn.Module, pattern: str,
                                     base_rank: int = 32,
                                     depth_ratio: float = 0.8) -> Dict[str, Tuple[int, int]]:
    """
    Assign ranks per MultiheadAttention layer based on architectural patterns.
    Useful when you want to manually override ranks before compression.

    Args:
        model: the model containing MultiheadAttention layers
        pattern: 'uniform', 'deeper_deeper', or 'deeper_shallower'
        base_rank: base rank (will be scaled)
        depth_ratio: how much ranks increase/decrease with depth (0.0 to 1.0)

    Returns:
        dict: {layer_name: (rank_qkv, rank_out)}
    """
    ranks = {}
    layers = [name for name, mod in model.named_modules() if isinstance(mod, nn.MultiheadAttention)]
    num_layers = len(layers)
    for i, name in enumerate(layers):
        if pattern == 'deeper_deeper':
            factor = 1 + depth_ratio * (i / max(1, num_layers - 1))
        elif pattern == 'deeper_shallower':
            factor = 1 + depth_ratio * (1 - i / max(1, num_layers - 1))
        else:  # uniform
            factor = 1.0
        rank = int(np.clip(base_rank * factor, 4, 128))
        ranks[name] = (rank, rank)
    return ranks