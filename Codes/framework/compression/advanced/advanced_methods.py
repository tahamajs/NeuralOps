"""
AdvancedZeroShotCompressor: 14 pre‑defined compression strategies.
Registered as 'advanced_zero_shot'.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple
from .core.base import BaseCompressor
from .core.registry import COMPRESSOR_REGISTRY
from ..attentions.compressed_attention import CompressedAttentionLayer


@COMPRESSOR_REGISTRY.register("advanced_zero_shot")
class AdvancedZeroShotCompressor(BaseCompressor):
    def __init__(self, method: str = "standard_svd", base_threshold: float = 0.90,
                 min_rank: int = 4, **kwargs):
        super().__init__(model=None, name="advanced_zero_shot", **kwargs)
        self.method = method
        self.base_threshold = base_threshold
        self.min_rank = min_rank

    def _get_rank_from_energy(self, S, threshold):
        if len(S) == 0:
            return self.min_rank
        energy = np.cumsum(S) / np.sum(S)
        try:
            rank = np.argmax(energy >= threshold) + 1
        except ValueError:
            rank = self.min_rank
        return max(self.min_rank, min(rank, len(S)))

    def compress(self, model: nn.Module = None) -> Tuple[nn.Module, Dict]:
        if model is not None:
            self.model = model
        print(f"🔧 Starting Method: {self.method}")
        compression_stats = []
        orig_total_params = sum(p.numel() for p in self.model.parameters())

        # Collect all MultiheadAttention layers
        layers_to_compress = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                w_qkv = module.in_proj_weight.data.cpu().numpy()
                w_o = module.out_proj.weight.data.cpu().numpy()
                _, S_qkv, _ = np.linalg.svd(w_qkv, full_matrices=False)
                _, S_out, _ = np.linalg.svd(w_o, full_matrices=False)
                layers_to_compress.append({
                    'name': name,
                    'module': module,
                    'S_qkv': S_qkv,
                    'S_out': S_out,
                    'original_dim': module.embed_dim,
                    'index': len(layers_to_compress)
                })

        total_layers = len(layers_to_compress)
        if total_layers == 0:
            self._compressed_params = orig_total_params
            return self.model, self.get_compression_stats()

        # Determine ranks per layer based on method
        ranks_list = self._compute_ranks(layers_to_compress, total_layers)

        # Replace layers
        compressed_count = 0
        for i, (info, (r_qkv, r_out)) in enumerate(zip(layers_to_compress, ranks_list)):
            try:
                new_layer = CompressedAttentionLayer(info['module'], r_qkv, r_out, self.method)
                parent_name = '.'.join(info['name'].split('.')[:-1])
                child_name = info['name'].split('.')[-1]
                parent = self.model.get_submodule(parent_name) if parent_name else self.model
                setattr(parent, child_name, new_layer)
                compressed_count += 1

                orig_p = sum(p.numel() for p in info['module'].parameters())
                new_p = sum(p.numel() for p in new_layer.parameters())
                compression_stats.append({
                    'layer': info['name'],
                    'original_params': orig_p,
                    'compressed_params': new_p,
                    'ratio': new_p / orig_p,
                    'saved': orig_p - new_p,
                    'target_rank': r_qkv
                })
            except Exception as e:
                print(f"   ❌ Failed to compress {info['name']}: {e}")

        new_total_params = sum(p.numel() for p in self.model.parameters())
        saved = orig_total_params - new_total_params
        reduction_pct = (saved / orig_total_params) * 100
        print(f"✅ Compression Complete! ({self.method})")
        print(f"   Layers: {compressed_count}/{total_layers}, Reduction: {reduction_pct:.2f}%")

        self._compressed_params = new_total_params
        return self.model, self.get_compression_stats()

    def _compute_ranks(self, layers, total_layers):
        ranks = []
        if self.method == "standard_svd":
            ranks = [(self._get_rank_from_energy(info['S_qkv'], self.base_threshold),
                      self._get_rank_from_energy(info['S_out'], self.base_threshold))
                     for info in layers]
        elif self.method == "head_pruning":
            half_dim = self.min_rank * 2
            ranks = [(half_dim, half_dim) for _ in layers]
        elif self.method == "layer_skipping":
            skip_ratio = 0.3
            skip_count = int(total_layers * skip_ratio)
            for i, info in enumerate(layers):
                if i < total_layers - skip_count:
                    r = self._get_rank_from_energy(info['S_qkv'], self.base_threshold)
                else:
                    r = self.min_rank
                ranks.append((r, r))
        elif self.method == "quantized_svd":
            ranks = [(self._get_rank_from_energy(info['S_qkv'], 0.75),
                      self._get_rank_from_energy(info['S_out'], 0.75))
                     for info in layers]
        elif self.method == "blockwise_lowrank":
            block_size = 4
            for i, info in enumerate(layers):
                block_idx = i // block_size
                r = max(self.min_rank, int(self._get_rank_from_energy(info['S_qkv'], self.base_threshold) * (1 - 0.1 * block_idx)))
                ranks.append((r, r))
        elif self.method == "randomized_svd":
            fixed_rank = 16
            ranks = [(fixed_rank, fixed_rank) for _ in layers]
        elif self.method == "tensor_train_decomp":
            start_r = self._get_rank_from_energy(layers[0]['S_qkv'], self.base_threshold)
            end_r = max(self.min_rank, int(start_r * 0.2))
            for i, info in enumerate(layers):
                progress = i / (total_layers - 1) if total_layers > 1 else 0
                r = int(start_r * (end_r / start_r) ** (progress * 2))
                r = max(self.min_rank, min(r, info['original_dim']))
                ranks.append((r, r))
        elif self.method == "cp_decomposition":
            fixed_cp_rank = 6
            ranks = [(fixed_cp_rank, fixed_cp_rank) for _ in layers]
        elif self.method == "dynamic_token_pruning":
            for i, info in enumerate(layers):
                r = self._get_rank_from_energy(info['S_qkv'], self.base_threshold)
                if i > total_layers * 0.6:
                    r = max(self.min_rank, int(r * 0.6))
                ranks.append((r, r))
        elif self.method == "spectral_clustering_prune":
            median_energy = np.median([np.mean(l['S_qkv']) for l in layers])
            for info in layers:
                if np.mean(info['S_qkv']) < median_energy:
                    r = self.min_rank
                else:
                    r = self._get_rank_from_energy(info['S_qkv'], self.base_threshold)
                ranks.append((r, r))
        elif self.method == "hash_quantization":
            fixed_hash_rank = 8
            ranks = [(fixed_hash_rank, fixed_hash_rank) for _ in layers]
        elif self.method == "moe_sparsification":
            for i, info in enumerate(layers):
                if i < total_layers // 2:
                    r = self._get_rank_from_energy(info['S_qkv'], self.base_threshold)
                else:
                    r = max(self.min_rank, int(self._get_rank_from_energy(info['S_qkv'], self.base_threshold) * 0.3))
                ranks.append((r, r))
        elif self.method == "hybrid_advanced":
            mid = total_layers // 2
            deep_start = int(total_layers * 0.7)
            for i, info in enumerate(layers):
                if i < mid:
                    start_r = self._get_rank_from_energy(layers[0]['S_qkv'], self.base_threshold)
                    end_r = max(self.min_rank, int(start_r * 0.2))
                    progress = i / (mid - 1) if mid > 1 else 0
                    r = int(start_r * (end_r / start_r) ** (progress * 2))
                elif i < deep_start:
                    r = self._get_rank_from_energy(info['S_qkv'], self.base_threshold)
                else:
                    r = max(self.min_rank, int(r * 0.3))
                ranks.append((r, r))
        else:
            raise ValueError(f"Unknown method: {self.method}")
        return ranks