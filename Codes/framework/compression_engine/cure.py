"""
CUR‑MoE: Interpretable CUR decomposition for MoE expert layers.
Registered as 'cur_moe'.
"""
from typing import Any

import torch
import torch.nn as nn
from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY


def cur_compress(weight, rank) -> Any:
    """
    Approximate weight matrix W as C @ U @ R, where C and R are column/row subsets.
    """
    out_dim, in_dim = weight.shape
    # Column and row selection (random for simplicity; real CUR uses leverage scores)
    col_idx = torch.randperm(in_dim)[:rank]
    row_idx = torch.randperm(out_dim)[:rank]
    C = weight[:, col_idx]  # (out, rank)
    R = weight[row_idx, :]  # (rank, in)
    # Solve for U: C @ U @ R ≈ W  ->  U ≈ C^+ @ W @ R^+
    C_pinv = torch.linalg.pinv(C.float())
    R_pinv = torch.linalg.pinv(R.float())
    U = C_pinv @ weight.float() @ R_pinv  # (rank, rank)
    return (C @ U @ R).to(weight.dtype)



def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    """Safely extract configuration attribute from None, dict, or object."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class CURMoECompressor(BaseCompressor):
    """
    Compresses expert layers using CUR decomposition.

    Args:
        model: PyTorch model.
        rank: rank for CUR decomposition (default 32).
    """

    def __init__(self, model, rank=32, target_attention_only=False, **kwargs):
        super().__init__(model, name="cur_moe", **kwargs)
        self.rank = rank
        self.target_attention_only = target_attention_only

    def compress(self) -> Any:
        """Compress the target module using this strategy."""
        for name, module in self.model.named_modules():
            target_mod = getattr(module, 'base_layer', module)
            if self.target_attention_only and not any(k in name.lower() for k in ["attn", "attention", "q_proj", "k_proj", "v_proj", "out_proj", "o_proj", "q_lin", "k_lin", "v_lin", "out_lin", "query", "key", "value", "c_attn", "c_proj"]):
                for p in module.parameters(): p.requires_grad = False
                continue
            if isinstance(target_mod, nn.Linear) and "expert" in name:
                target_mod.weight.data = cur_compress(target_mod.weight.data, self.rank)
                if hasattr(module, 'base_layer'):
                    if hasattr(module, 'lora_A'):
                        pass
                else:
                    for p in module.parameters(): p.requires_grad = False
        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()


@COMPRESSOR_REGISTRY.register("cur_moe")
def cur_moe_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """Cur moe factory."""
    rank = getattr(comp_config, "cur_moe_rank", kwargs.get("rank", 64))
    method = getattr(comp_config, "cur_moe_method", kwargs.get("method", "leverage"))
    compressor = CURMoECompressor(model, rank=rank, **kwargs)
    compressed_model, _ = compressor.compress()
    return compressed_model
