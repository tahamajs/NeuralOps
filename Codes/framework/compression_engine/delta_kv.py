"""
DeltaKV: Residual‑Based KV Cache Compression via Long‑Range Similarity.

Instead of discarding tokens, DeltaKV encodes semantic residuals relative
to retrieved historical references, preserving fidelity while reducing
storage to 29% of the original.

Paper: "DeltaKV: Residual-Based KV Cache Compression via Long-Range Similarity"
       Hao et al., arXiv:2602.08005.
Registered as 'delta_kv'.
"""

import torch
import torch.nn as nn
from framework.core.base import BaseCompressor
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, METHOD_REGISTRY


class DeltaKVStore:
    """
    Maintains a reference key‑value store and compresses new K/V tensors
    by storing only the residuals relative to the closest historical references.
    """
    def __init__(self, max_refs: int = 128, topk: int = 4,
                 compression_ratio: float = 0.29):
        self.max_refs = max_refs
        self.topk = topk
        self.compression_ratio = compression_ratio
        self.ref_k: torch.Tensor | None = None
        self.ref_v: torch.Tensor | None = None

    @torch.no_grad()
    def update(self, k: torch.Tensor, v: torch.Tensor):
        """Update the reference store with new K/V tensors."""
        B, H, N, D = k.shape
        if B == 0:
            return
        ks = k[0]   # (H, N, D)
        vs = v[0]
        if self.ref_k is None:
            self.ref_k = ks
            self.ref_v = vs
        else:
            self.ref_k = torch.cat([self.ref_k, ks], dim=1)
            self.ref_v = torch.cat([self.ref_v, vs], dim=1)
        # Truncate to max_refs
        if self.ref_k.size(1) > self.max_refs:
            self.ref_k = self.ref_k[:, -self.max_refs:, :]
            self.ref_v = self.ref_v[:, -self.max_refs:, :]

    @torch.no_grad()
    def compress(self, k: torch.Tensor, v: torch.Tensor):
        """
        Compress K/V by computing residuals relative to the closest reference
        tokens, keeping only the top residuals.
        """
        if self.ref_k is None:
            return k, v

        B, H, N, D = k.shape
        tk = min(self.topk, self.ref_k.size(1))

        # Compute similarities between current K and reference K
        sim = torch.matmul(k, self.ref_k.transpose(-2, -1))  # (B, H, N, C)
        _, idx = sim.topk(tk, dim=-1)                         # (B, H, N, tk)

        # Gather reference vectors
        idx_expand = idx.unsqueeze(-1).expand(-1, -1, -1, -1, D)
        ref_k_expanded = (self.ref_k.unsqueeze(0).unsqueeze(2)
                          .expand(B, H, N, -1, -1))
        ref_v_expanded = (self.ref_v.unsqueeze(0).unsqueeze(2)
                          .expand(B, H, N, -1, -1))

        rk = ref_k_expanded.gather(3, idx_expand).mean(dim=3)
        rv = ref_v_expanded.gather(3, idx_expand).mean(dim=3)

        # Residual
        k_residual = k - rk
        v_residual = v - rv

        # Keep only significant residuals (L2 norm > threshold)
        k_keep_mask = k_residual.norm(dim=-1) > 1e-3   # (B, H, N)
        v_keep_mask = v_residual.norm(dim=-1) > 1e-3

        # Apply masks: zero out negligible residuals
        k_compressed = torch.where(k_keep_mask.unsqueeze(-1), k_residual,
                                   torch.zeros_like(k_residual))
        v_compressed = torch.where(v_keep_mask.unsqueeze(-1), v_residual,
                                   torch.zeros_like(v_residual))

        return k_compressed, v_compressed


@COMPRESSOR_REGISTRY.register("delta_kv")
class DeltaKVCompressor(BaseCompressor):
    """
    Compresses attention projection weights using a DeltaKV‑inspired residual
    approach: decomposing weights into a low‑rank reference component plus
    sparse residuals.

    Args:
        model: PyTorch model.
        rank: rank of the reference (low‑rank) component (default 16).
        sparsity: fraction of residual entries to retain (default 0.29).
    """
    def __init__(self, model, rank=16, sparsity=0.29, **kwargs):
        super().__init__(model, name="delta_kv", **kwargs)
        self.rank = rank
        self.sparsity = sparsity

    def compress(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                W = module.weight.data.float()

                # Step 1: Low‑rank reference (SVD)
                U, S, Vt = torch.linalg.svd(W, full_matrices=False)
                r = min(self.rank, min(W.shape))
                L = (U[:, :r] * S[:r].unsqueeze(0)) @ Vt[:r, :]

                # Step 2: Residual
                residual = W - L

                # Step 3: Keep only top sparsity fraction of residual entries
                flat_r = residual.abs().view(-1)
                keep_n = max(1, int(flat_r.numel() * self.sparsity))
                threshold = torch.topk(flat_r, keep_n).values[-1]
                mask = (residual.abs() >= threshold).float()
                residual_sparse = residual * mask

                # Step 4: Final weight = reference + sparse residual
                W_compressed = L + residual_sparse
                module.weight.data = W_compressed.to(dtype=module.weight.dtype)

        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()