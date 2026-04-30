# studio/compression/adaptive_compressor.py
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple

from ..attentions.compressed_attention import CompressedAttentionLayer
from .patterns import get_layer_patterns, detect_architecture

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
        orig_total = sum(p.numel() for p in model.parameters())
        stats = []

        layers_to_compress = []
        for name, module in model.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                w_qkv = module.in_proj_weight.data.cpu().numpy()
                w_o = module.out_proj.weight.data.cpu().numpy()
                _, S_qkv, _ = np.linalg.svd(w_qkv, full_matrices=False)
                _, S_out, _ = np.linalg.svd(w_o, full_matrices=False)
                layers_to_compress.append({
                    'name': name, 'module': module,
                    'S_qkv': S_qkv, 'S_out': S_out,
                    'original_dim': module.embed_dim,
                })

        total = len(layers_to_compress)
        if total == 0:
            return model, [], 0.0

        # Determine ranks per layer based on method (simplified – see full code in advanced_methods.py)
        ranks_list = []
        for info in layers_to_compress:
            r_qkv = self._get_rank_from_energy(info['S_qkv'], self.base_threshold)
            r_out = self._get_rank_from_energy(info['S_out'], self.base_threshold)
            ranks_list.append((r_qkv, r_out))

        compressed_count = 0
        for info, (r_qkv, r_out) in zip(layers_to_compress, ranks_list):
            try:
                new_layer = CompressedAttentionLayer(info['module'], r_qkv, r_out, method)
                parent = model.get_submodule('.'.join(info['name'].split('.')[:-1])) if '.' in info['name'] else model
                setattr(parent, info['name'].split('.')[-1], new_layer)
                compressed_count += 1
                stats.append({'layer': info['name'], 'original_params': sum(p.numel() for p in info['module'].parameters()),
                              'compressed_params': sum(p.numel() for p in new_layer.parameters())})
            except Exception as e:
                print(f"   ❌ Failed {info['name']}: {e}")

        new_total = sum(p.numel() for p in model.parameters())
        reduction = (orig_total - new_total) / orig_total * 100
        return model, stats, reduction

class AdaptiveSVDCompressor:
    def __init__(self, energy_threshold=0.95, min_rank=16):
        self.energy_threshold = energy_threshold
        self.min_rank = min_rank

    def compress_model(self, model):
        # Simplified version – compresses MultiheadAttention layers
        for name, mod in model.named_modules():
            if isinstance(mod, nn.MultiheadAttention):
                compressed = CompressedAttentionLayer(mod, self.energy_threshold, self.min_rank)
                parent = model.get_submodule('.'.join(name.split('.')[:-1])) if '.' in name else model
                setattr(parent, name.split('.')[-1], compressed)
        return model