"""
MoDeGPT: modular decomposition of transformer blocks.
Decomposes pairs of linear layers (e.g., Q and K) jointly.
Registered as 'modegpt'.
"""

import torch
import torch.nn as nn
import numpy as np
from framework.core.base import BaseCompressor
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, METHOD_REGISTRY


class MoDeGPTCompressor(BaseCompressor):
    """
    Compresses transformer blocks by decomposing pairs of attention projections
    (e.g., Q & K, V & Output) using a shared low‑rank structure.

    Args:
        model: PyTorch model.
        rank: rank for the decomposition.
        method: 'cr' or 'cur' (placeholder; current implementation uses SVD).
    """
    def __init__(self, model, rank=64, method='cr', **kwargs):
        super().__init__(model, name="modegpt", **kwargs)
        self.rank = rank
        self.method = method

    def compress(self):
        # Identify pairs of Linear layers in transformer blocks
        # A simple heuristic: look for 'q_lin' and 'k_lin' in the same parent module.
        for name, module in self.model.named_modules():
            if hasattr(module, 'q_lin') and hasattr(module, 'k_lin'):
                q_lin = module.q_lin
                k_lin = module.k_lin
                # Decompose jointly: concatenate weights and apply SVD
                W_q = q_lin.weight.data.float()
                W_k = k_lin.weight.data.float()
                W_joint = torch.cat([W_q, W_k], dim=0)   # (2*H, D)
                U, S, Vt = torch.linalg.svd(W_joint, full_matrices=False)
                r = min(self.rank, min(W_joint.shape))
                U_r = U[:, :r] * S[:r].unsqueeze(0)
                Vt_r = Vt[:r, :]
                W_approx = U_r @ Vt_r
                W_q_approx = W_approx[:W_q.shape[0], :]
                W_k_approx = W_approx[W_q.shape[0]:, :]
                q_lin.weight.data = W_q_approx.to(dtype=q_lin.weight.dtype)
                k_lin.weight.data = W_k_approx.to(dtype=k_lin.weight.dtype)
        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()
    

@COMPRESSOR_REGISTRY.register("modegpt")
def modegpt_factory(model, comp_config):
    rank = getattr(comp_config, 'modegpt_rank', 64)
    method = getattr(comp_config, 'modegpt_method', 'cr')
    compressor = MoDeGPTCompressor(model, rank=rank, method=method)
    compressed_model, _ = compressor.compress()
    return compressed_model

