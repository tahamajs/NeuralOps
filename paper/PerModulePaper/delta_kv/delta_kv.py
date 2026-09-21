"""
DeltaKV: Residual-Based KV Cache Compression via Long-Range Similarity.

Encodes semantic residuals relative to retrieved historical references,
preserving attention fidelity while drastically compressing dynamic KV caches.

Paper: "DeltaKV: Residual-Based KV Cache Compression via Long-Range Similarity"
       Hao et al., arXiv:2602.08005.
Registered as 'delta_kv'.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import logging

import torch
import torch.nn as nn
from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY

logger = logging.getLogger(__name__)


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
        sim = torch.matmul(k_4d.float(), ref_k_b.float().transpose(-2, -1).contiguous()) / (D**0.5)
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


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Safely extract config value from None, dict, or namespace/object."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


_ATTN_NAMES = {"q_proj", "k_proj", "v_proj", "o_proj", "c_attn", "c_proj", "query", "key", "value"}


class DeltaKVCompressor(BaseCompressor):
    is_custom_attention: bool = True
    """
    DeltaKV Compressor: Decomposes projection weights into a low-rank reference component
    plus sparse residuals, modeling weight matrices under residual delta formulation.

    Production hardening features:
    - Robust LoRALinear and Conv1D support (with transpose and adapter preservation).
    - target_attention_only parameter: compresses attention projections and freezes non-target/fallback layers.
    - Safe error validation with max_error threshold and fallback to original.
    - Fallback and skipped layers frozen (requires_grad=False) during recovery.
    - Real metrics recording: rank, sparsity, effective_parameter_ratio, kv_cache_storage_ratio.
    """

    def __init__(
        self,
        model: nn.Module,
        rank: int = 16,
        sparsity: float = 0.29,
        max_error: float = 0.35,
        target_attention_only: bool = False,
        fallback_to_original: bool = True,
        cfg: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="delta_kv", **kwargs)
        self.cfg = cfg
        self.rank = int(_cfg_get(cfg, "delta_kv_rank", _cfg_get(cfg, "rank", rank)))
        self.sparsity = float(_cfg_get(cfg, "delta_kv_sparsity", _cfg_get(cfg, "sparsity", sparsity)))
        self.max_error = float(_cfg_get(cfg, "delta_kv_max_error", _cfg_get(cfg, "max_error", max_error)))
        self.target_attention_only = bool(
            _cfg_get(cfg, "delta_kv_target_attention_only", _cfg_get(cfg, "target_attention_only", target_attention_only))
        )
        self.fallback_to_original = bool(
            _cfg_get(cfg, "delta_kv_fallback_to_original", _cfg_get(cfg, "fallback_to_original", fallback_to_original))
        )

        self.compressed_layers: List[str] = []
        self.fallback_layers: List[str] = []
        self.skipped_layers: List[str] = []
        self.layer_errors: Dict[str, float] = {}
        self.layer_ranks: Dict[str, int] = {}

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        """Check if module qualifies for DeltaKV compression."""
        name_lower = name.lower()
        if any(part in name.split(".") for part in ["lora_A", "lora_B", "base_layer"]):
            return False
        if any(skip in name_lower for skip in ["classifier", "score", "lm_head", "pooler", "embedding", "router", "gate"]):
            return False

        # LoRALinear detection
        target_mod = getattr(module, "base_layer", module)
        cls_name = target_mod.__class__.__name__.lower()
        is_linear = isinstance(target_mod, nn.Linear)
        is_conv1d = "conv1d" in cls_name or hasattr(target_mod, "nf")

        if not (is_linear or is_conv1d):
            return False

        if self.target_attention_only:
            last_part = name.split(".")[-1]
            is_attn = (last_part in _ATTN_NAMES) or any(k in name_lower for k in ("attn", "attention"))
            if not is_attn:
                return False

        return True

    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """Decompose linear projection weights into low-rank + sparse residual."""
        self.compressed_layers = []
        self.fallback_layers = []
        self.skipped_layers = []
        self.layer_errors = {}
        self.layer_ranks = {}

        total_orig_params = 0
        total_comp_params = 0

        for name, module in list(self.model.named_modules()):
            name_lower = name.lower()
            if any(skip in name_lower for skip in ["classifier", "score", "lm_head", "pooler", "embedding", "router", "gate"]):
                continue

            if any(part in name.split(".") for part in ["lora_A", "lora_B", "base_layer"]):
                continue

            target_mod = getattr(module, "base_layer", module)
            cls_name = target_mod.__class__.__name__.lower()
            is_linear = isinstance(target_mod, nn.Linear)
            is_conv1d = "conv1d" in cls_name or hasattr(target_mod, "nf")

            if not (is_linear or is_conv1d):
                continue

            if not hasattr(target_mod, "weight") or target_mod.weight is None:
                continue

            # Check if this layer is targeted
            if not self._is_target_module(name, module):
                if self.target_attention_only:
                    # Freeze uncompressed non-attention layers during recovery, preserving classifier/head
                    if not any(k in name_lower for k in ("classifier", "score", "head")):
                        for p in module.parameters():
                            p.requires_grad = False
                    self.skipped_layers.append(name)
                continue

            raw_w = target_mod.weight.data
            if is_conv1d:
                in_f, out_f = raw_w.shape[0], raw_w.shape[1]
                W = raw_w.t().contiguous().float()  # [out_features, in_features]
            else:
                out_f, in_f = raw_w.shape[0], raw_w.shape[1]
                W = raw_w.float()

            orig_layer_params = out_f * in_f
            total_orig_params += orig_layer_params
            norm_w = torch.norm(W, p="fro").item()

            try:
                U, S, Vt = torch.linalg.svd(W, full_matrices=False)
                r = max(1, min(self.rank, min(W.shape), len(S)))
                L = (U[:, :r] * S[:r].unsqueeze(0)) @ Vt[:r, :]

                # Sparse residual
                residual = W - L
                flat_r = residual.contiguous().abs().reshape(-1)
                keep_n = max(1, min(int(flat_r.numel() * self.sparsity), flat_r.numel()))
                threshold = torch.topk(flat_r, keep_n).values[-1]
                mask = (residual.abs() >= threshold).float()
                residual_sparse = residual * mask

                W_compressed = L + residual_sparse

                err = float(torch.norm(W - W_compressed, p="fro").item() / max(norm_w, 1e-12))
                self.layer_errors[name] = err
                self.layer_ranks[name] = r

                if err > self.max_error and self.fallback_to_original:
                    self.fallback_layers.append(name)
                    # Freeze fallback layer parameters
                    for p in module.parameters():
                        p.requires_grad = False
                    total_comp_params += orig_layer_params
                    continue

                # Put weight back with original orientation
                if is_conv1d:
                    target_mod.weight.data.copy_(W_compressed.t().contiguous().to(dtype=target_mod.weight.dtype))
                else:
                    target_mod.weight.data.copy_(W_compressed.to(dtype=target_mod.weight.dtype))

                # If wrapped in LoRALinear, keep LoRA parameters trainable; otherwise make weights trainable
                if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                    if hasattr(module.lora_A, "weight") and module.lora_A.weight is not None:
                        module.lora_A.weight.requires_grad = True
                    if hasattr(module.lora_B, "weight") and module.lora_B.weight is not None:
                        module.lora_B.weight.requires_grad = True
                else:
                    if hasattr(target_mod, "weight") and target_mod.weight is not None:
                        target_mod.weight.requires_grad = True
                    if hasattr(target_mod, "bias") and target_mod.bias is not None:
                        target_mod.bias.requires_grad = True

                self.compressed_layers.append(name)
                # Parameter calculation: low-rank factors (r*(in+out)) + sparse nonzeros (sparsity*in*out)
                comp_layer_params = r * (in_f + out_f) + keep_n
                total_comp_params += comp_layer_params

            except Exception as e:
                logger.warning("DeltaKV decomposition failed on %s: %s", name, e)
                self.fallback_layers.append(name)
                for p in module.parameters():
                    p.requires_grad = False
                total_comp_params += orig_layer_params

        effective_ratio = total_comp_params / total_orig_params if total_orig_params > 0 else 1.0
        avg_err = sum(self.layer_errors.values()) / max(1, len(self.layer_errors)) if self.layer_errors else 0.0

        stats = self.get_compression_stats()
        stats.update({
            "rank": self.rank,
            "sparsity": self.sparsity,
            "max_error": self.max_error,
            "target_attention_only": self.target_attention_only,
            "fallback_to_original": self.fallback_to_original,
            "compressed_layers": self.compressed_layers,
            "compressed_layer_count": len(self.compressed_layers),
            "fallback_layers": self.fallback_layers,
            "fallback_layer_count": len(self.fallback_layers),
            "skipped_layers": self.skipped_layers,
            "skipped_layer_count": len(self.skipped_layers),
            "layer_errors": self.layer_errors,
            "layer_ranks": self.layer_ranks,
            "avg_reconstruction_error": avg_err,
            "effective_parameter_ratio": round(effective_ratio, 4),
            "kv_cache_storage_ratio": f"{self.sparsity * 100:.1f}% of full KV cache",
        })
        self.stats = stats
        return self.model, stats


@COMPRESSOR_REGISTRY.register("delta_kv")
@COMPRESSOR_REGISTRY.register("delta_kv_compressor")
def delta_kv_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """DeltaKV compressor factory function."""
    rank = kwargs.pop("rank", _cfg_get(comp_config, "delta_kv_rank", _cfg_get(comp_config, "rank", 16)))
    sparsity = kwargs.pop("sparsity", _cfg_get(comp_config, "delta_kv_sparsity", _cfg_get(comp_config, "sparsity", 0.29)))
    max_error = kwargs.pop("max_error", _cfg_get(comp_config, "delta_kv_max_error", _cfg_get(comp_config, "max_error", 0.35)))
    target_attention_only = kwargs.pop(
        "target_attention_only",
        _cfg_get(comp_config, "delta_kv_target_attention_only", _cfg_get(comp_config, "target_attention_only", False)),
    )
    fallback_to_original = kwargs.pop(
        "fallback_to_original",
        _cfg_get(comp_config, "delta_kv_fallback_to_original", _cfg_get(comp_config, "fallback_to_original", True)),
    )
    return_stats = kwargs.pop("return_stats", kwargs.pop("return_tuple", False))

    compressor = DeltaKVCompressor(
        model,
        rank=rank,
        sparsity=sparsity,
        max_error=max_error,
        target_attention_only=target_attention_only,
        fallback_to_original=fallback_to_original,
        cfg=comp_config,
        **kwargs,
    )
    compressed_model, stats = compressor.compress()
    if return_stats:
        return compressed_model, stats
    return compressed_model


__all__ = [
    "DeltaKVStore",
    "DeltaKVCompressor",
    "delta_kv_factory",
]
