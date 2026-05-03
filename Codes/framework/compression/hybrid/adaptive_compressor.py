"""
Adaptive Transformer Compressor: automatically compresses all
nn.MultiheadAttention layers using energy‑threshold SVD.
"""

import torch.nn as nn
from .core.base import BaseCompressor
from .core.registry import COMPRESSOR_REGISTRY
from ..attentions.compressed_attention import CompressedAttentionLayer


@COMPRESSOR_REGISTRY.register("adaptive_svd")
class AdaptiveTransformerCompressor(BaseCompressor):
    def __init__(self, model, energy_threshold=0.95, min_rank=16, **kwargs):
        super().__init__(model, name="adaptive_svd", **kwargs)
        self.energy_threshold = energy_threshold
        self.min_rank = min_rank

    def compress(self):
        print(f"🔧 Starting Compression (Threshold: {self.energy_threshold}, "
              f"Min Rank: {self.min_rank})")
        compression_stats = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                try:
                    compressed = CompressedAttentionLayer(
                        module, self.energy_threshold, self.min_rank
                    )
                    # Replace in place
                    parent_name = '.'.join(name.split('.')[:-1])
                    child_name = name.split('.')[-1]
                    parent = self.model.get_submodule(parent_name) if parent_name else self.model
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

        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()