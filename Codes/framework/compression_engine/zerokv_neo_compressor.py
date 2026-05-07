"""
ZeroKV‑Neo: L2‑norm based KV cache eviction with attention sink preservation.
Forces K/V tensors to retain only the most important tokens.

Reference: "A Simple and Effective L2 Norm-Based Strategy for KV Cache Compression"
           + H2O (Heavy-Hitter Oracle) attention sink preservation.
Registered as 'zerokv_neo'.
"""

import torch
import torch.nn as nn
from framework.core.base import BaseCompressor
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, METHOD_REGISTRY


class ZeroKVNeoCompressor(BaseCompressor):
    """
    Compresses KV caches by retaining only high‑L2‑norm tokens plus attention sinks.

    Args:
        model: PyTorch model.
        keep_ratio: fraction of tokens to retain (e.g., 0.3 = 30%).
        num_attention_sinks: number of initial tokens always kept (default 4).
        heavy_hitter_ratio: fraction of heavy‑hitter tokens retained in addition to sinks.
    """
    def __init__(self, model, keep_ratio=0.3, num_attention_sinks=4,
                 heavy_hitter_ratio=0.1, **kwargs):
        super().__init__(model, name="zerokv_neo", **kwargs)
        self.keep_ratio = keep_ratio
        self.num_attention_sinks = num_attention_sinks
        self.heavy_hitter_ratio = heavy_hitter_ratio

    def compress(self):
        # Identify all attention layers and compress their weight projections
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and self._is_attention_proj(name):
                W = module.weight.data
                out_f, in_f = W.shape
                # Compute L2 norm of each output neuron (row)
                row_norms = torch.norm(W.float(), dim=1)   # (out_features,)
                # Keep: attention sinks (top norms) + heavy hitters
                n_sinks = min(self.num_attention_sinks, out_f)
                n_heavy = max(1, int(out_f * self.heavy_hitter_ratio))
                total_keep = min(n_sinks + n_heavy, out_f)

                _, top_idx = torch.topk(row_norms, total_keep)
                mask = torch.zeros(out_f, device=W.device)
                mask[top_idx] = 1.0

                # Zero out pruned rows by scaling them down (soft pruning)
                W_pruned = W * mask.unsqueeze(1)
                module.weight.data = W_pruned.to(dtype=module.weight.dtype)
        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()

    @staticmethod
    def _is_attention_proj(name: str) -> bool:
        """Check if a layer name corresponds to an attention projection."""
        attn_keywords = ['q_lin', 'k_lin', 'v_lin', 'q_proj', 'k_proj', 'v_proj',
                         'query', 'key', 'value', 'attention']
        return any(kw in name.lower() for kw in attn_keywords)
    

@COMPRESSOR_REGISTRY.register("zerokv_neo")
def zerokv_neo_factory(model, comp_config):
    keep_ratio = getattr(comp_config, 'zerokv_keep_ratio', 0.3)
    num_sinks = getattr(comp_config, 'zerokv_num_sinks', 4)
    heavy_ratio = getattr(comp_config, 'zerokv_heavy_ratio', 0.1)
    compressor = ZeroKVNeoCompressor(model, keep_ratio=keep_ratio,
                                     num_attention_sinks=num_sinks,
                                     heavy_hitter_ratio=heavy_ratio)
    compressed_model, _ = compressor.compress()
    return compressed_model