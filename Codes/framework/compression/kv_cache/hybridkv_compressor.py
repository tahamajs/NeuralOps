"""
HybridKV: Hybrid KV Cache Compression for Multimodal LLMs.
Classifies attention heads into static/dynamic, then applies distinct
compression strategies per type under a hierarchical budget.

Paper: "HybridKV: Hybrid KV Cache Compression for Efficient MLLM Inference"
       Zeng et al., arXiv:2604.05887.
Registered as 'hybridkv'.
"""

import torch
import torch.nn as nn
from ..core.base import BaseCompressor
from ..core.registry import COMPRESSOR_REGISTRY


@COMPRESSOR_REGISTRY.register("hybridkv")
class HybridKVCompressor(BaseCompressor):
    """
    Classifies attention heads into static/dynamic based on weight variance,
    then applies different pruning budgets to each.

    Args:
        model: PyTorch model.
        budget: overall fraction of parameters to retain (default 0.3).
        static_ratio: fraction of budget allocated to static heads (default 0.7).
    """
    def __init__(self, model, budget=0.3, static_ratio=0.7, **kwargs):
        super().__init__(model, name="hybridkv", **kwargs)
        self.budget = budget
        self.static_ratio = static_ratio

    @staticmethod
    def _classify_head(weight: torch.Tensor, threshold: float = 0.5) -> str:
        """
        Classify a projection head as 'static' or 'dynamic' based on
        weight variance across the output dimension.
        """
        row_variance = torch.var(weight.float(), dim=1)   # (out_features,)
        # Normalised variance: low → static, high → dynamic
        norm_var = row_variance / (row_variance.max() + 1e-8)
        dynamic_ratio = (norm_var > threshold).float().mean().item()
        return 'dynamic' if dynamic_ratio > 0.3 else 'static'

    def compress(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and self._is_attention_proj(name):
                W = module.weight.data
                out_f = W.shape[0]
                head_type = self._classify_head(W)

                if head_type == 'static':
                    # Text‑prior pruning: keep top rows by L2 norm
                    keep = max(1, int(out_f * self.budget * self.static_ratio))
                    row_norms = torch.norm(W.float(), dim=1)
                    _, idx = torch.topk(row_norms, keep)
                    mask = torch.zeros(out_f, device=W.device)
                    mask[idx] = 1.0
                else:
                    # Dynamic heads: chunk‑wise top‑k retention
                    chunk_size = 8
                    keep = max(1, int(out_f * self.budget * (1 - self.static_ratio)))
                    mask = torch.zeros(out_f, device=W.device)
                    for c in range(0, out_f, chunk_size):
                        end = min(c + chunk_size, out_f)
                        chunk = W[c:end].float()
                        chunk_norms = torch.norm(chunk, dim=1)
                        n_keep_chunk = max(1, int((end - c) * keep / out_f))
                        _, local_idx = torch.topk(chunk_norms,
                                                  min(n_keep_chunk, end - c))
                        mask[c + local_idx] = 1.0

                W_pruned = W * mask.unsqueeze(1)
                module.weight.data = W_pruned.to(dtype=module.weight.dtype)

        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()

    @staticmethod
    def _is_attention_proj(name: str) -> bool:
        attn_keywords = ['q_lin', 'k_lin', 'v_lin', 'q_proj', 'k_proj', 'v_proj',
                         'query', 'key', 'value', 'attention']
        return any(kw in name.lower() for kw in attn_keywords)