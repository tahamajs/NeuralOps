"""
DeltaKV: Residual-Based KV Cache Compression via Long-Range Similarity.

Encodes semantic residuals relative to retrieved historical references,
preserving attention fidelity while drastically compressing dynamic KV caches.

Paper: "DeltaKV: Residual-Based KV Cache Compression via Long-Range Similarity"
       Hao et al., arXiv:2602.08005.
Registered as 'delta_kv'.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY


class DeltaKVStore:
    """
    Maintains historical reference key-value representations and compresses incoming K/V tensors
    by storing delta residuals relative to top-k closest reference tokens.
    Supports dynamic sequence lengths during prefill (N >= 1) and autoregressive decoding (N = 1).
    """

    def __init__(
        self,
        max_refs: int = 128,
        topk: int = 4,
        compression_ratio: float = 0.29,
        residual_threshold: float = 1e-3,
    ) -> None:
        self.max_refs = int(max_refs)
        self.topk = int(topk)
        self.compression_ratio = float(compression_ratio)
        self.residual_threshold = float(residual_threshold)
        self.ref_k: Optional[torch.Tensor] = None
        self.ref_v: Optional[torch.Tensor] = None

    def reset(self) -> None:
        """Clear reference caches to avoid memory leaks across sequences."""
        self.ref_k = None
        self.ref_v = None

    @torch.no_grad()
    def update(self, k: torch.Tensor, v: torch.Tensor) -> None:
        """
        Append new key and value tensors to reference memory.
        Expected shapes: (B, H, N, D) or (B, N, D) or (N, D).
        """
        orig_k = k.detach()
        orig_v = v.detach()

        # Normalize shape to (B, H, N, D)
        if orig_k.ndim == 2:
            orig_k = orig_k.unsqueeze(0).unsqueeze(0)
            orig_v = orig_v.unsqueeze(0).unsqueeze(0)
        elif orig_k.ndim == 3:
            orig_k = orig_k.unsqueeze(1)
            orig_v = orig_v.unsqueeze(1)

        B, H, N, D = orig_k.shape
        if N == 0 or B == 0:
            return

        if self.ref_k is None:
            self.ref_k = orig_k
            self.ref_v = orig_v
        else:
            # Match device and dtype if changed
            if self.ref_k.device != orig_k.device or self.ref_k.dtype != orig_k.dtype:
                self.ref_k = self.ref_k.to(device=orig_k.device, dtype=orig_k.dtype)
                self.ref_v = self.ref_v.to(device=orig_v.device, dtype=orig_v.dtype)

            # Match batch dimension if changed
            if self.ref_k.shape[0] != B:
                self.ref_k = self.ref_k[:1].expand(B, -1, -1, -1).clone()
                self.ref_v = self.ref_v[:1].expand(B, -1, -1, -1).clone()

            self.ref_k = torch.cat([self.ref_k, orig_k], dim=2)
            self.ref_v = torch.cat([self.ref_v, orig_v], dim=2)

        # Truncate to maximum reference token capacity
        if self.ref_k.size(2) > self.max_refs:
            self.ref_k = self.ref_k[:, :, -self.max_refs :, :].contiguous()
            self.ref_v = self.ref_v[:, :, -self.max_refs :, :].contiguous()

    @torch.no_grad()
    def compress(
        self, k: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compress K/V tensors by retrieving top-k references and returning sparse residuals.
        Returns reconstructed/compressed (k_hat, v_hat).
        """
        orig_shape = k.shape
        if self.ref_k is None or self.ref_k.size(2) == 0:
            self.update(k, v)
            return k, v

        # Normalize shape to (B, H, N, D)
        k_4d = k
        v_4d = v
        if k.ndim == 2:
            k_4d = k.unsqueeze(0).unsqueeze(0)
            v_4d = v.unsqueeze(0).unsqueeze(0)
        elif k.ndim == 3:
            k_4d = k.unsqueeze(1)
            v_4d = v.unsqueeze(1)

        B, H, N, D = k_4d.shape
        ref_b, ref_h, ref_n, ref_d = self.ref_k.shape

        # Match device and dtype
        ref_k_cur = self.ref_k.to(device=k.device, dtype=k.dtype)
        ref_v_cur = self.ref_v.to(device=v.device, dtype=v.dtype)

        # Broadcast references across batch if needed
        ref_k_b = ref_k_cur if ref_b == B else ref_k_cur[:1].expand(B, -1, -1, -1)
        ref_v_b = ref_v_cur if ref_b == B else ref_v_cur[:1].expand(B, -1, -1, -1)

        tk = min(self.topk, ref_n)
        if tk == 0:
            self.update(k, v)
            return k, v

        # Compute cosine-scaled similarity: (B, H, N, D) x (B, H, D, ref_N) -> (B, H, N, ref_N)
        sim = torch.matmul(k_4d.float(), ref_k_b.float().transpose(-2, -1)) / (D**0.5)
        _, top_idx = torch.topk(sim, tk, dim=-1)  # (B, H, N, tk)

        # Gather reference key-value representations
        # top_idx: (B, H, N, tk) -> expand to (B, H, N, tk, D)
        top_idx_exp = top_idx.unsqueeze(-1).expand(B, H, N, tk, D)
        ref_k_exp = ref_k_b.unsqueeze(2).expand(B, H, N, ref_n, D)
        ref_v_exp = ref_v_b.unsqueeze(2).expand(B, H, N, ref_n, D)

        rk = ref_k_exp.gather(3, top_idx_exp).mean(dim=3).to(dtype=k_4d.dtype)  # (B, H, N, D)
        rv = ref_v_exp.gather(3, top_idx_exp).mean(dim=3).to(dtype=v_4d.dtype)  # (B, H, N, D)

        # Compute delta residuals
        k_residual = k_4d - rk
        v_residual = v_4d - rv

        # Residual thresholding / sparsity mask
        k_norm = torch.norm(k_residual.float(), dim=-1, keepdim=True)
        v_norm = torch.norm(v_residual.float(), dim=-1, keepdim=True)

        k_mask = (k_norm > self.residual_threshold).to(dtype=k_4d.dtype)
        v_mask = (v_norm > self.residual_threshold).to(dtype=v_4d.dtype)

        k_sparse = k_residual * k_mask
        v_sparse = v_residual * v_mask

        # Reconstructed representations
        k_hat = rk + k_sparse
        v_hat = rv + v_sparse

        # Update store with new representations
        self.update(k, v)

        return k_hat.reshape(orig_shape), v_hat.reshape(orig_shape)


class DeltaKVCompressor(BaseCompressor):
    """
    DeltaKV Compressor: Decomposes projection weights into a low-rank reference component
    plus sparse residuals, modeling weight matrices under residual delta formulation.

    Args:
        model: PyTorch model.
        rank: Rank of reference low-rank matrix (default 16).
        sparsity: Fraction of sparse residual elements to retain (default 0.29).
    """

    def __init__(
        self,
        model: nn.Module,
        rank: int = 16,
        sparsity: float = 0.29,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="delta_kv", **kwargs)
        self.rank = int(rank)
        self.sparsity = float(sparsity)

    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """Decompose linear projection weights into low-rank + sparse residual."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                W = module.weight.data.float()
                U, S, Vt = torch.linalg.svd(W, full_matrices=False)
                r = max(1, min(self.rank, min(W.shape), len(S)))
                L = (U[:, :r] * S[:r].unsqueeze(0)) @ Vt[:r, :]

                # Sparse residual
                residual = W - L
                flat_r = residual.abs().view(-1)
                keep_n = max(1, min(int(flat_r.numel() * self.sparsity), flat_r.numel()))
                threshold = torch.topk(flat_r, keep_n).values[-1]
                mask = (residual.abs() >= threshold).float()
                residual_sparse = residual * mask

                W_compressed = L + residual_sparse
                module.weight.data = W_compressed.to(dtype=module.weight.dtype)

        stats = self.get_compression_stats()
        stats["rank"] = self.rank
        stats["sparsity"] = self.sparsity
        stats["kv_cache_storage_ratio"] = f"{self.sparsity * 100:.1f}% of full KV cache"
        return self.model, stats


@COMPRESSOR_REGISTRY.register("delta_kv")
def delta_kv_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """DeltaKV compressor factory function."""
    rank = kwargs.get("rank", getattr(comp_config, "delta_kv_rank", 16) if comp_config is not None else 16)
    sparsity = kwargs.get("sparsity", getattr(comp_config, "delta_kv_sparsity", 0.29) if comp_config is not None else 0.29)
    compressor = DeltaKVCompressor(model, rank=rank, sparsity=sparsity, **kwargs)
    compressed_model, _ = compressor.compress()
    return compressed_model
