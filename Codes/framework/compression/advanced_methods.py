import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path

# Import the compressed attention layer (from earlier)
from ..attentions.compressed_attention import CompressedAttentionLayer

class AdvancedZeroShotCompressor:
    def __init__(self, method: str, base_threshold: float = 0.90, min_rank: int = 4):
        self.method = method
        self.base_threshold = base_threshold
        self.min_rank = min_rank
        
    def _get_rank_from_energy(self, S, threshold):
        if len(S) == 0:
            return self.min_rank
        energy = np.cumsum(S) / np.sum(S)
        try:
            rank = np.argmax(energy >= threshold) + 1
        except:
            rank = self.min_rank
        return max(self.min_rank, min(rank, len(S)))

    def compress_model(self, model: nn.Module, method: str) -> Tuple[nn.Module, List[Dict], float]:
        print(f"🔧 Starting Method: {method}")
        compression_stats = []
        orig_total_params = sum(p.numel() for p in model.parameters())
        
        # Collect all MultiheadAttention layers
        layers_to_compress = []
        for name, module in model.named_modules():
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
            return model, [], 0.0

        # Determine ranks per layer based on method
        ranks_list = []
        if method == "standard_svd":
            ranks_list = [(self._get_rank_from_energy(info['S_qkv'], self.base_threshold),
                           self._get_rank_from_energy(info['S_out'], self.base_threshold)) 
                          for info in layers_to_compress]
        elif method == "head_pruning":
            half_dim = self.min_rank * 2
            ranks_list = [(half_dim, half_dim) for _ in layers_to_compress]
        elif method == "layer_skipping":
            skip_ratio = 0.3
            skip_count = int(total_layers * skip_ratio)
            for i, info in enumerate(layers_to_compress):
                if i < total_layers - skip_count:
                    r = self._get_rank_from_energy(info['S_qkv'], self.base_threshold)
                else:
                    r = self.min_rank
                ranks_list.append((r, r))
        elif method == "quantized_svd":
            ranks_list = [(self._get_rank_from_energy(info['S_qkv'], 0.75),
                           self._get_rank_from_energy(info['S_out'], 0.75)) 
                          for info in layers_to_compress]
        elif method == "blockwise_lowrank":
            block_size = 4
            for i, info in enumerate(layers_to_compress):
                block_idx = i // block_size
                r = max(self.min_rank, int(self._get_rank_from_energy(info['S_qkv'], self.base_threshold) * (1 - 0.1 * block_idx)))
                ranks_list.append((r, r))
        elif method == "randomized_svd":
            fixed_rank = 16
            ranks_list = [(fixed_rank, fixed_rank) for _ in layers_to_compress]
        elif method == "tensor_train_decomp":
            start_r = self._get_rank_from_energy(layers_to_compress[0]['S_qkv'], self.base_threshold)
            end_r = max(self.min_rank, int(start_r * 0.2))
            for i, info in enumerate(layers_to_compress):
                progress = i / (total_layers - 1) if total_layers > 1 else 0
                r = int(start_r * (end_r / start_r) ** (progress * 2))
                r = max(self.min_rank, min(r, info['original_dim']))
                ranks_list.append((r, r))
        elif method == "cp_decomposition":
            fixed_cp_rank = 6
            ranks_list = [(fixed_cp_rank, fixed_cp_rank) for _ in layers_to_compress]
        elif method == "dynamic_token_pruning":
            for i, info in enumerate(layers_to_compress):
                r = self._get_rank_from_energy(info['S_qkv'], self.base_threshold)
                if i > total_layers * 0.6:
                    r = max(self.min_rank, int(r * 0.6))
                ranks_list.append((r, r))
        elif method == "spectral_clustering_prune":
            median_energy = np.median([np.mean(l['S_qkv']) for l in layers_to_compress])
            for info in layers_to_compress:
                if np.mean(info['S_qkv']) < median_energy:
                    r = self.min_rank
                else:
                    r = self._get_rank_from_energy(info['S_qkv'], self.base_threshold)
                ranks_list.append((r, r))
        elif method == "hash_quantization":
            fixed_hash_rank = 8
            ranks_list = [(fixed_hash_rank, fixed_hash_rank) for _ in layers_to_compress]
        elif method == "moe_sparsification":
            for i, info in enumerate(layers_to_compress):
                if i < total_layers // 2:
                    r = self._get_rank_from_energy(info['S_qkv'], self.base_threshold)
                else:
                    r = max(self.min_rank, int(self._get_rank_from_energy(info['S_qkv'], self.base_threshold) * 0.3))
                ranks_list.append((r, r))
        elif method == "hybrid_advanced":
            mid = total_layers // 2
            deep_start = int(total_layers * 0.7)
            for i, info in enumerate(layers_to_compress):
                if i < mid:
                    start_r = self._get_rank_from_energy(layers_to_compress[0]['S_qkv'], self.base_threshold)
                    end_r = max(self.min_rank, int(start_r * 0.2))
                    progress = i / (mid - 1) if mid > 1 else 0
                    r = int(start_r * (end_r / start_r) ** (progress * 2))
                elif i < deep_start:
                    r = self._get_rank_from_energy(info['S_qkv'], self.base_threshold)
                else:
                    r = max(self.min_rank, int(r * 0.3))
                ranks_list.append((r, r))
        else:
            raise ValueError(f"Unknown method: {method}")

        # Replace layers
        compressed_count = 0
        for i, (info, (r_qkv, r_out)) in enumerate(zip(layers_to_compress, ranks_list)):
            try:
                new_layer = CompressedAttentionLayer(info['module'], r_qkv, r_out, method)
                # Replace in model
                parent_name = '.'.join(info['name'].split('.')[:-1])
                child_name = info['name'].split('.')[-1]
                parent = model.get_submodule(parent_name) if parent_name else model
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

        new_total_params = sum(p.numel() for p in model.parameters())
        saved = orig_total_params - new_total_params
        reduction_pct = (saved / orig_total_params) * 100
        print(f"✅ Compression Complete! ({method})")
        print(f"   Layers: {compressed_count}/{total_layers}, Reduction: {reduction_pct:.2f}%")
        return model, compression_stats, reduction_pct