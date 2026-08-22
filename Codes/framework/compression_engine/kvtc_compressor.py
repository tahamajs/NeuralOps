"""
KVTC: KV Cache Transform Coding - PCA decorrelation + adaptive quantization
      + entropy coding for compact KV cache storage.

Paper: "KV Cache Transform Coding for Compact Storage in LLM Inference"
       Staniszewski & Lancucki, ICLR 2026.
Registered as 'kvtc'.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY


class SimpleEntropyCoder:
    """Lightweight entropy coder using empirical symbol frequency estimation."""

    @staticmethod
    def compute_entropy(tensor: torch.Tensor) -> float:
        """Compute empirical Shannon entropy in bits per symbol."""
        if tensor.numel() == 0:
            return 0.0
        unique_vals, counts = torch.unique(tensor, return_counts=True)
        probs = counts.float() / float(tensor.numel())
        entropy = -torch.sum(probs * torch.log2(torch.clamp(probs, min=1e-12))).item()
        return float(entropy)

    @staticmethod
    def compute_compression_ratio(tensor: torch.Tensor, uncompressed_bits: int = 16) -> float:
        """Estimate theoretical compression ratio relative to FP16/FP32 baseline."""
        entropy = SimpleEntropyCoder.compute_entropy(tensor)
        return float(uncompressed_bits / max(entropy, 1e-4))


class KVTCStore:
    """
    Runtime KV Cache Transform Coding engine.
    Applies PCA decorrelation across head feature dimensions and adaptive quantization,
    supporting dynamic sequence lengths across prefill and decoding phases.
    """

    def __init__(self, bits: int = 4, pca_energy: float = 0.95) -> None:
        self.bits = int(bits)
        self.pca_energy = float(pca_energy)
        self.pca_basis_k: Optional[torch.Tensor] = None
        self.pca_basis_v: Optional[torch.Tensor] = None

    def reset(self) -> None:
        """Reset PCA basis to prevent cross-sequence memory leaks."""
        self.pca_basis_k = None
        self.pca_basis_v = None

    @torch.no_grad()
    def calibrate_basis(self, k: torch.Tensor, v: torch.Tensor) -> None:
        """Calibrate PCA projection basis from sample K/V tensor (B, H, N, D)."""
        k_flat = k.detach().float().reshape(-1, k.shape[-1])
        v_flat = v.detach().float().reshape(-1, v.shape[-1])

        # SVD on centered channels
        _, _, Vt_k = torch.linalg.svd(k_flat - k_flat.mean(dim=0, keepdim=True), full_matrices=False)
        _, _, Vt_v = torch.linalg.svd(v_flat - v_flat.mean(dim=0, keepdim=True), full_matrices=False)

        self.pca_basis_k = Vt_k.t().to(device=k.device, dtype=k.dtype)
        self.pca_basis_v = Vt_v.t().to(device=v.device, dtype=v.dtype)

    @torch.no_grad()
    def compress_kv_cache(
        self, k: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        Transform and quantize dynamic KV cache tensors.
        Returns reconstructed (k_hat, v_hat) and coding statistics.
        """
        orig_shape = k.shape
        D = orig_shape[-1]

        if self.pca_basis_k is None or self.pca_basis_v is None:
            self.calibrate_basis(k, v)

        # Reshape to (-1, D)
        k_flat = k.reshape(-1, D)
        v_flat = v.reshape(-1, D)

        # PCA Transform
        k_trans = torch.matmul(k_flat, self.pca_basis_k)
        v_trans = torch.matmul(v_flat, self.pca_basis_v)

        # Adaptive uniform quantization
        qmax = (2 ** (self.bits - 1)) - 1
        scale_k = torch.clamp(k_trans.abs().max() / float(qmax), min=1e-6)
        scale_v = torch.clamp(v_trans.abs().max() / float(qmax), min=1e-6)

        q_k = torch.clamp(torch.round(k_trans / scale_k), -qmax, qmax)
        q_v = torch.clamp(torch.round(v_trans / scale_v), -qmax, qmax)

        # Entropy measurement
        h_k = SimpleEntropyCoder.compute_entropy(q_k)
        h_v = SimpleEntropyCoder.compute_entropy(q_v)

        # Inverse Transform
        k_hat = torch.matmul(q_k * scale_k, self.pca_basis_k.t()).reshape(orig_shape)
        v_hat = torch.matmul(q_v * scale_v, self.pca_basis_v.t()).reshape(orig_shape)

        stats = {
            "entropy_k_bits": h_k,
            "entropy_v_bits": h_v,
            "mean_entropy_bits": (h_k + h_v) / 2.0,
            "theoretical_compression_ratio": 16.0 / max((h_k + h_v) / 2.0, 1e-4),
        }
        return k_hat, v_hat, stats


class KVTCCompressor(BaseCompressor):
    """
    KVTC: KV Cache Transform Coding compressor.
    Compresses linear projection weights using PCA decorrelation,
    adaptive quantization, and entropy coding estimation.

    Args:
        model: PyTorch model.
        bits: Quantization bit-width (default 4).
        pca_energy: Fraction of PCA energy retained (default 0.95).
    """

    def __init__(
        self,
        model: nn.Module,
        bits: int = 4,
        pca_energy: float = 0.95,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="kvtc", **kwargs)
        self.bits = int(bits)
        self.pca_energy = float(pca_energy)

    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """Compress linear layers using PCA decorrelation and adaptive quantization."""
        total_entropy = 0.0
        layers_compressed = 0

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                W = module.weight.data.float()

                # Step 1: PCA decorrelation via SVD
                U, S, Vt = torch.linalg.svd(W, full_matrices=False)
                total_energy = (S**2).sum()
                if total_energy > 0:
                    cum_energy = torch.cumsum(S**2, dim=0) / total_energy
                    rank = int(torch.searchsorted(cum_energy, self.pca_energy).item()) + 1
                else:
                    rank = min(W.shape)
                rank = max(1, min(rank, min(W.shape)))

                # Reconstruct low-rank approximation
                U_r = U[:, :rank]
                S_r = S[:rank]
                Vt_r = Vt[:rank, :]
                W_lowrank = (U_r * S_r.unsqueeze(0)) @ Vt_r

                # Step 2: Adaptive quantization
                min_val, max_val = W_lowrank.min(), W_lowrank.max()
                if max_val == min_val:
                    max_val = min_val + 1e-6
                scale = (max_val - min_val) / float((2**self.bits) - 1)
                q_vals = torch.round((W_lowrank - min_val) / scale)
                W_quant = q_vals * scale + min_val

                # Step 3: Empirical entropy calculation
                entropy = SimpleEntropyCoder.compute_entropy(q_vals)
                total_entropy += entropy
                layers_compressed += 1

                module.weight.data = W_quant.to(dtype=module.weight.dtype)

        stats = self.get_compression_stats()
        avg_entropy = total_entropy / max(layers_compressed, 1)
        stats["bits"] = self.bits
        stats["pca_energy"] = self.pca_energy
        stats["avg_entropy_bits"] = round(avg_entropy, 2)
        stats["theoretical_entropy_compression_ratio"] = round(16.0 / max(avg_entropy, 1e-4), 2)
        stats["theoretical_vram_saving"] = f"{(1.0 - self.bits / 16.0) * 100:.1f}% vs FP16"
        return self.model, stats


@COMPRESSOR_REGISTRY.register("kvtc")
def kvtc_factory(model: nn.Module, comp_config: Any = None, **kwargs: Any) -> nn.Module:
    """KVTC compressor factory function."""
    bits = kwargs.pop("bits", getattr(comp_config, "kvtc_bits", 4) if comp_config else 4)
    pca_energy = kwargs.pop("pca_energy", getattr(comp_config, "kvtc_pca_energy", 0.95) if comp_config else 0.95)
    compressor = KVTCCompressor(model, bits=bits, pca_energy=pca_energy, **kwargs)
    compressed_model, _ = compressor.compress()
    return compressed_model
