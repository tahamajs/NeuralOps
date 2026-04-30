import torch
import torch.nn as nn
import numpy as np
from ..attentions.compressed_attention import CompressedAttentionLayer

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