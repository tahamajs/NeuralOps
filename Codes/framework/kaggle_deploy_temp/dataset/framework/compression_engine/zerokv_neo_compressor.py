"""ZeroKV‑Neo: L2‑norm based KV cache eviction with attention sink preservation.
Registered as 'zerokv_neo'."""

from typing import Any

import torch
import torch.nn as nn
from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY


class ZeroKVNeoCompressor(BaseCompressor):
    def __init__(
        self,
        model,
        keep_ratio=0.3,
        num_attention_sinks=4,
        heavy_hitter_ratio=0.1,
        **kwargs
    ):
        super().__init__(model, name="zerokv_neo", **kwargs)
        self.keep_ratio = keep_ratio
        self.num_attention_sinks = num_attention_sinks
        self.heavy_hitter_ratio = heavy_hitter_ratio

    def compress(self) -> Any:
        """Compress the target module using this strategy."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and self._is_attention_proj(name):
                W = module.weight.data
                out_f, in_f = W.shape

                # Total neurons to keep based on keep_ratio
                total_keep = max(1, int(out_f * self.keep_ratio))

                # Compute L2 norm of each output neuron
                row_norms = torch.norm(W.float(), dim=1)

                # We want to keep attention sinks (top few) and some heavy hitters,
                # but the combined number cannot exceed total_keep.
                n_sinks = min(self.num_attention_sinks, total_keep)
                n_heavy = min(
                    max(1, int(out_f * self.heavy_hitter_ratio)), total_keep - n_sinks
                )
                # Select the top norms for sinks and then for heavy hitters
                _, top_sink_idx = torch.topk(row_norms, n_sinks, largest=True)
                # Remove sinks from consideration for heavy hitters
                mask_all = torch.ones(out_f, dtype=torch.bool, device=W.device)
                mask_all[top_sink_idx] = False
                remaining_norms = row_norms[mask_all]
                if n_heavy > 0 and remaining_norms.numel() > 0:
                    _, top_heavy_idx_relative = torch.topk(
                        remaining_norms, n_heavy, largest=True
                    )
                    # Map back to original indices
                    remaining_indices = torch.where(mask_all)[0]
                    top_heavy_idx = remaining_indices[top_heavy_idx_relative]
                else:
                    top_heavy_idx = torch.tensor([], device=W.device, dtype=torch.long)

                keep_idx = torch.cat([top_sink_idx, top_heavy_idx])
                # Build mask
                mask = torch.zeros(out_f, device=W.device)
                mask[keep_idx] = 1.0

                # Prune weights and biases
                module.weight.data = W * mask.unsqueeze(1)
                if module.bias is not None:
                    module.bias.data = module.bias.data * mask

        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()

    @staticmethod
    def _is_attention_proj(name: str) -> bool:
        """ is attention proj."""
        attn_keywords = [
            "q_lin",
            "k_lin",
            "v_lin",
            "q_proj",
            "k_proj",
            "v_proj",
            "query",
            "key",
            "value",
            "attention",
        ]
        return any(kw in name.lower() for kw in attn_keywords)


@COMPRESSOR_REGISTRY.register("zerokv_neo")
def zerokv_neo_factory(model, comp_config) -> Any:
    """Zerokv neo factory."""
    keep_ratio = getattr(comp_config, "zerokv_keep_ratio", 0.3)
    num_sinks = getattr(comp_config, "zerokv_num_sinks", 4)
    heavy_ratio = getattr(comp_config, "zerokv_heavy_ratio", 0.1)
    compressor = ZeroKVNeoCompressor(
        model,
        keep_ratio=keep_ratio,
        num_attention_sinks=num_sinks,
        heavy_hitter_ratio=heavy_ratio,
    )
    compressed_model, _ = compressor.compress()
    return compressed_model
